# imports
import random
import logging
import time
import os
from dataclasses import dataclass
from itertools import product
from multiprocessing import cpu_count, Process, Queue, Value, Array, Event
from typing import List, Optional, Tuple, Any, Dict, Union
import math
import argparse
import csv
from pathlib import Path
import sys
import gc
from GenAIRR.alleles import AlleleTypes
from GenAIRR.pipeline import AugmentationPipeline
from GenAIRR.steps import AugmentationStep
from GenAIRR.dataconfig import DataConfig
from GenAIRR.steps import (
    SimulateSequence, FixVPositionAfterTrimmingIndexAmbiguity, FixDPositionAfterTrimmingIndexAmbiguity,
    FixJPositionAfterTrimmingIndexAmbiguity, CorrectForVEndCut, CorrectForDTrims, CorruptSequenceBeginning,
    InsertNs, InsertIndels, ShortDValidation, DistillMutationRate
)
# Import Uniform without importing S5F/pandas-heavy modules
try:
    from GenAIRR.mutation.uniform import Uniform  # type: ignore
except Exception:
    # Fallback if direct module import path differs
    from GenAIRR.mutation import Uniform  # type: ignore
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Training Dataset Generator')

# Global flag derived from DataConfig at runtime
has_d = False

# Per-process cached state (not shared across processes)
_WORKER_STATE: Dict[str, Any] = {
    'dataconfig': None,
    'has_d': False,
    'pipeline': None,
    'sim_step': None,
    'pipelines': {},  # dc_key -> {'dc': DataConfig, 'pipeline': AugmentationPipeline, 'sim_step': SimulateSequence, 'chain_type': str}
}


class _silence_stdio:
    """Context manager to silence stdout/stderr temporarily."""
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        devnull = open(os.devnull, 'w')
        self._devnull = devnull
        sys.stdout = devnull
        sys.stderr = devnull
        return self
    def __exit__(self, exc_type, exc, tb):
        try:
            sys.stdout = self._stdout
            sys.stderr = self._stderr
        finally:
            try:
                self._devnull.close()
            except Exception:
                pass

def get_simulation_pipeline(has_d: bool) -> List[AugmentationStep]:
    """Build the simulation pipeline, excluding D-specific steps for light chains."""
    simulation_steps: List[AugmentationStep] = [
        SimulateSequence(
            Uniform(min_mutation_rate=0.003, max_mutation_rate=0.2, productive=True),
            specific_v=None,
            specific_d=None,
            specific_j=None,
            productive=True,
        ),
        FixVPositionAfterTrimmingIndexAmbiguity(),
        FixJPositionAfterTrimmingIndexAmbiguity(),
        CorrectForVEndCut(),
        CorruptSequenceBeginning(0.7, [0.4, 0.4, 0.2], 576, 210, 310, 50),
        InsertNs(0.02, 0.5),
        InsertIndels(0.5, 5, 0.5, 0.5),
        DistillMutationRate(),
    ]
    if has_d:
        # Insert D-related steps in appropriate places
        # After V fix, add D fix
        simulation_steps.insert(2, FixDPositionAfterTrimmingIndexAmbiguity())
        # After V-end correction, add D trims correction
        simulation_steps.insert(4, CorrectForDTrims())
        # Before Indels, add short D validation
        simulation_steps.insert(6, ShortDValidation())
    return simulation_steps

def generate_all_allele_combinations(v_alleles: List[Any], j_alleles: List[Any], d_alleles: Optional[List[Any]]) -> List[Tuple[Optional[Any], Optional[Any], Optional[Any]]]:
    """
    Generate all possible combinations, returned as uniform (V, D, J) tuples.
    - If D is not provided: only VJ combos -> (V, None, J)
    - If D is provided: VD -> (V, D, None), DJ -> (None, D, J), and VDJ -> (V, D, J)
    This uniform tuple form simplifies worker logic.
    """
    if d_alleles is None:
        vj = [(v, None, j) for v, j in product(v_alleles, j_alleles)]
        return vj
    else:
        vd = [(v, d, None) for v, d in product(v_alleles, d_alleles)]
        dj = [(None, d, j) for d, j in product(d_alleles, j_alleles)]
        vdj = [(v, d, j) for v, d, j in product(v_alleles, d_alleles, j_alleles)]
        return vd + dj + vdj


def load_genairr_dataconfig(config_path: str) -> DataConfig:
    """
    Load a GenAIRR DataConfig from a given path.
    """
    # check if its a name of a dataconfig in the package
    from GenAIRR.data import _CONFIG_NAMES
    if config_path in _CONFIG_NAMES:
        # import the dataconfig from the package by doing from GenAIRR.data import <config_path>
        module = __import__('GenAIRR.data', fromlist=[config_path])
        dataconfig = getattr(module, config_path)
    else:
        import pickle
        with open(config_path, 'rb') as f:
            dataconfig = pickle.load(f)
    return dataconfig
        

def get_alleles(dataconfig: DataConfig, repeats:int=1) -> Dict[str, List[str]]:
    V_alleles = [i for j in dataconfig.v_alleles for i in dataconfig.v_alleles[j]]
    J_alleles = [i for j in dataconfig.j_alleles for i in dataconfig.j_alleles[j]]
    result = {
        'V': V_alleles,
        'J': J_alleles,
       
    }
    if dataconfig.metadata.has_d:
        D_alleles = [i for j in dataconfig.d_alleles for i in dataconfig.d_alleles[j]]
        result['D'] = D_alleles
        
    # apply repeats
    for key in result:
        result[key] = result[key] * repeats
        
    # calculate number of combinations combinatorially including repeats of the sum of all vj combinations or vd+dj+vdj combinations
    if 'D' in result:
        num_combinations = len(result['V']) * len(result['D']) + len(result['D']) * len(result['J']) + len(result['V']) * len(result['D']) * len(result['J'])
    else:
        num_combinations = len(result['V']) * len(result['J'])

    logger.info('  The Number of Sequence Combinations is: {:,}'.format(num_combinations))

    return result

def _init_worker(dataconfig_name_or_path: str):
    """Initializer per process: load dataconfig, bind to steps, build pipeline."""
    # Reduce logging in workers to avoid cluttering the UI
    logging.basicConfig(level=logging.WARNING)
    with _silence_stdio():
        dc = load_genairr_dataconfig(dataconfig_name_or_path)
    _WORKER_STATE['dataconfig'] = dc
    _WORKER_STATE['has_d'] = getattr(dc.metadata, 'has_d', False)
    # Bind dataconfig to all AugmentationSteps
    AugmentationStep.set_dataconfig(dc)
    steps = get_simulation_pipeline(_WORKER_STATE['has_d'])
    pipe = AugmentationPipeline(steps=steps)
    _WORKER_STATE['pipeline'] = pipe
    _WORKER_STATE['sim_step'] = steps[0]
    # Noisy logs disabled for cleaner UI




def _to_record(obj: Any) -> Dict[str, Any]:
    """Normalize simulation output to a dict suitable for CSV writing."""
    if hasattr(obj, 'get_dict'):
        rec = obj.get_dict()
    else:
        rec = obj
    if isinstance(rec, list):
        if len(rec) == 1 and isinstance(rec[0], dict):
            rec = rec[0]
    if not isinstance(rec, dict):
        raise RuntimeError("Simulation output is not a dict-like record")
    return rec

def _worker_emit_records(
    dataconfig_name_or_path: str,
    worker_id: int,
    alleles_to_simulate: List[Tuple[Optional[Any], Optional[Any], Optional[Any]]],
    out_queue: Queue,
    per_worker_progress: Any,
    vj_seq_counter: Any,
    vd_seq_counter: Any,
    dj_seq_counter: Any,
    vdj_seq_counter: Any,
    error_counter: Any,
):
    """Simulate records for each allele tuple and push dicts to a queue for a single writer."""
    # Initialize per-process state (safe under spawn)
    _init_worker(dataconfig_name_or_path)
    pipe = _WORKER_STATE['pipeline']
    sim_step: SimulateSequence = _WORKER_STATE['sim_step']
    for v_allele, d_allele, j_allele in alleles_to_simulate:
        # Assign desired targets
        sim_step.specific_v = v_allele
        sim_step.specific_d = d_allele
        sim_step.specific_j = j_allele

        # Toggle productive & simulate twice with retries/fallback
        for productive in (True, False):
            sim_step.productive = productive
            um = getattr(sim_step, 'mutation_model', None)
            if um is not None:
                um.productive = productive
                # Save original rates for fallback adjustments
                orig_min = getattr(um, 'min_mutation_rate', None)
                orig_max = getattr(um, 'max_mutation_rate', None)

            # Try up to 3 times, reducing mutation rate on failures, final fallback to 0 mutation
            last_exc: Optional[Exception] = None
            for attempt in range(3):
                try:
                    with _silence_stdio():
                        result = pipe.execute()
                    break
                except Exception as e:
                    last_exc = e
                    # Reduce mutation rate progressively if possible
                    if um is not None and orig_min is not None and orig_max is not None:
                        new_max = max(0.0, orig_max * (0.5 ** (attempt + 1)))
                        new_min = min(orig_min, new_max)
                        um.max_mutation_rate = new_max
                        um.min_mutation_rate = new_min
                    with error_counter.get_lock():
                        error_counter.value += 1
            else:
                # Final fallback: try with zero mutation if still failing
                if um is not None and orig_min is not None and orig_max is not None:
                    um.max_mutation_rate = 0.0
                    um.min_mutation_rate = 0.0
                try:
                    with _silence_stdio():
                        result = pipe.execute()
                except Exception:
                    # Give up for this (productive/non-productive) record; continue to next
                    # Restore original mutation rates and move on
                    if um is not None and orig_min is not None and orig_max is not None:
                        um.min_mutation_rate = orig_min
                        um.max_mutation_rate = orig_max
                    continue

            # Restore original mutation rates after success
            if um is not None and orig_min is not None and orig_max is not None:
                um.min_mutation_rate = orig_min
                um.max_mutation_rate = orig_max

            rec = _to_record(result)
            rec['productive'] = productive
            # Always include target fields to keep CSV schema stable
            rec['target_v'] = getattr(v_allele, 'name', str(v_allele)) if v_allele is not None else None
            rec['target_d'] = getattr(d_allele, 'name', str(d_allele)) if d_allele is not None else None
            rec['target_j'] = getattr(j_allele, 'name', str(j_allele)) if j_allele is not None else None
            out_queue.put(rec)
        # Update category sequence counters (+2 sequences per combo)
        if v_allele is not None and d_allele is None and j_allele is not None:
            with vj_seq_counter.get_lock():
                vj_seq_counter.value += 2
        elif v_allele is not None and d_allele is not None and j_allele is None:
            with vd_seq_counter.get_lock():
                vd_seq_counter.value += 2
        elif v_allele is None and d_allele is not None and j_allele is not None:
            with dj_seq_counter.get_lock():
                dj_seq_counter.value += 2
        elif v_allele is not None and d_allele is not None and j_allele is not None:
            with vdj_seq_counter.get_lock():
                vdj_seq_counter.value += 2
        # Per-worker progress (+1 combo)
        with per_worker_progress.get_lock():
            per_worker_progress[worker_id] += 1
    # Signal worker completion
    out_queue.put(None)


def _ensure_pipeline_for_dc(dc_name_or_path: str) -> Tuple[Any, SimulateSequence, str, DataConfig]:
    """Create or reuse a cached pipeline for a given dataconfig identifier in the current worker process."""
    cache = _WORKER_STATE.get('pipelines')
    if dc_name_or_path in cache:
        entry = cache[dc_name_or_path]
        return entry['pipeline'], entry['sim_step'], entry['chain_type'], entry['dc']
    # Not cached: load and build
    with _silence_stdio():
        dc = load_genairr_dataconfig(dc_name_or_path)
    chain_type = getattr(dc.metadata, 'chain_type', None)
    has_d = getattr(dc.metadata, 'has_d', False)
    AugmentationStep.set_dataconfig(dc)
    steps = get_simulation_pipeline(has_d)
    pipe = AugmentationPipeline(steps=steps)
    sim_step = steps[0]
    entry = {'dc': dc, 'pipeline': pipe, 'sim_step': sim_step, 'chain_type': chain_type}
    cache[dc_name_or_path] = entry
    return pipe, sim_step, chain_type, dc


def _worker_emit_records_multi(
    worker_id: int,
    tasks: List[Tuple[str, Optional[str], Tuple[Optional[Any], Optional[Any], Optional[Any]]]],
    out_queue: Queue,
    per_worker_progress: Any,
    vj_seq_counter: Any,
    vd_seq_counter: Any,
    dj_seq_counter: Any,
    vdj_seq_counter: Any,
    error_counter: Any,
):
    """Worker that handles multiple dataconfigs: each task contains (dc_name_or_path, chain_type_hint, (v,d,j))."""
    # Ensure cache exists
    if 'pipelines' not in _WORKER_STATE or _WORKER_STATE['pipelines'] is None:
        _WORKER_STATE['pipelines'] = {}

    for (dc_name_or_path, _chain_hint, (v_allele, d_allele, j_allele)) in tasks:
        pipe, sim_step, chain_type, dc = _ensure_pipeline_for_dc(dc_name_or_path)
        # Set current dataconfig on steps (paranoia; ensures correct context)
        AugmentationStep.set_dataconfig(dc)

        # Assign desired targets
        sim_step.specific_v = v_allele
        sim_step.specific_d = d_allele
        sim_step.specific_j = j_allele

        # Toggle productive & simulate twice with retries/fallback
        for productive in (True, False):
            sim_step.productive = productive
            um = getattr(sim_step, 'mutation_model', None)
            if um is not None:
                um.productive = productive
                # Save original rates for fallback adjustments
                orig_min = getattr(um, 'min_mutation_rate', None)
                orig_max = getattr(um, 'max_mutation_rate', None)

            last_exc: Optional[Exception] = None
            for attempt in range(3):
                try:
                    with _silence_stdio():
                        result = pipe.execute()
                    break
                except Exception as e:
                    last_exc = e
                    if um is not None and orig_min is not None and orig_max is not None:
                        new_max = max(0.0, orig_max * (0.5 ** (attempt + 1)))
                        new_min = min(orig_min, new_max)
                        um.max_mutation_rate = new_max
                        um.min_mutation_rate = new_min
                    with error_counter.get_lock():
                        error_counter.value += 1
            else:
                # Final fallback: try with zero mutation if still failing
                if um is not None and orig_min is not None and orig_max is not None:
                    um.max_mutation_rate = 0.0
                    um.min_mutation_rate = 0.0
                try:
                    with _silence_stdio():
                        result = pipe.execute()
                except Exception:
                    # Restore rates and continue
                    if um is not None and orig_min is not None and orig_max is not None:
                        um.min_mutation_rate = orig_min
                        um.max_mutation_rate = orig_max
                    continue

            if um is not None and orig_min is not None and orig_max is not None:
                um.min_mutation_rate = orig_min
                um.max_mutation_rate = orig_max

            rec = _to_record(result)
            rec['productive'] = productive
            rec['target_v'] = getattr(v_allele, 'name', str(v_allele)) if v_allele is not None else None
            rec['target_d'] = getattr(d_allele, 'name', str(d_allele)) if d_allele is not None else None
            rec['target_j'] = getattr(j_allele, 'name', str(j_allele)) if j_allele is not None else None
            rec['chain_type'] = str(chain_type) if chain_type is not None else None
            out_queue.put(rec)

        # Update category sequence counters (+2 sequences per combo)
        if v_allele is not None and d_allele is None and j_allele is not None:
            with vj_seq_counter.get_lock():
                vj_seq_counter.value += 2
        elif v_allele is not None and d_allele is not None and j_allele is None:
            with vd_seq_counter.get_lock():
                vd_seq_counter.value += 2
        elif v_allele is None and d_allele is not None and j_allele is not None:
            with dj_seq_counter.get_lock():
                dj_seq_counter.value += 2
        elif v_allele is not None and d_allele is not None and j_allele is not None:
            with vdj_seq_counter.get_lock():
                vdj_seq_counter.value += 2

        with per_worker_progress.get_lock():
            per_worker_progress[worker_id] += 1

    out_queue.put(None)


def _bar_line(label: str, current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        pct = 0.0
        fill = 0
        denom = 0
    else:
        pct = min(1.0, current / total)
        fill = int(pct * width)
        denom = total
    bar = '=' * fill + '.' * (width - fill)
    return f"{label:8} [{bar}] {current:,}/{denom:,} ({pct*100:5.1f}%)"


def _monitor_process(
    per_worker_progress: Any,
    worker_totals: Any,
    vj_seq_counter: Any,
    vd_seq_counter: Any,
    dj_seq_counter: Any,
    vdj_seq_counter: Any,
    error_counter: Any,
    vj_target: int,
    vd_target: int,
    dj_target: int,
    vdj_target: int,
    total_combos: int,
    start_time_val: Any,
    done_event: Any,
    refresh_interval: float = 0.5,
):
    try:
        # Hide cursor
        try:
            sys.stdout.write("\x1b[?25l")
            sys.stdout.flush()
        except Exception:
            pass
        while not done_event.is_set():
            # Clear screen to avoid clutter (ANSI + Windows cls for safety)
            try:
                os.system('')  # Enable ANSI in Windows terminals
                sys.stdout.write("\x1b[2J\x1b[H")
                sys.stdout.flush()
                # Fallback clear for Windows PowerShell
                os.system('cls')
            except Exception:
                os.system('cls')
            combos_done = sum(per_worker_progress[:])
            seq_done = combos_done * 2
            total_sequences = total_combos * 2

            elapsed = time.time() - start_time_val.value
            rate = seq_done / elapsed if elapsed > 0 else 0.0
            remaining_seq = max(total_sequences - seq_done, 0)
            eta = remaining_seq / rate if rate > 0 else float('inf')

            print("GenAIRR Dataset Generation Progress\n")
            print(_bar_line('Overall', combos_done, total_combos, 40))
            print(f"Sequences: {seq_done:,}/{total_sequences:,} | Rate: {rate:,.0f}/s | Elapsed: {elapsed:,.1f}s | ETA: {'∞' if eta==float('inf') else f'{eta:,.1f}s'} | Errors: {error_counter.value:,}\n")

            # Category stats
            print("By category (sequences):")
            print(_bar_line('VJ', vj_seq_counter.value, vj_target))
            print(_bar_line('VD', vd_seq_counter.value, vd_target))
            print(_bar_line('DJ', dj_seq_counter.value, dj_target))
            print(_bar_line('VDJ', vdj_seq_counter.value, vdj_target))
            print()

            # Per worker
            print("Workers:")
            for i in range(len(worker_totals)):
                curr = per_worker_progress[i]
                tot = worker_totals[i]
                print(_bar_line(f"W{i}", curr, tot))

            # Avoid flushing too frequently
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        pass
    finally:
        # Show cursor again
        try:
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()
        except Exception:
            pass


def _writer_process(output_csv: str, in_queue: Queue, n_workers: int, flush_every: int = 5000):
    """Single writer: consume dicts from queue and write to CSV without collisions."""
    header_written = False
    writer = None
    buffer: List[Dict[str, Any]] = []
    completed = 0
    # Ensure parent dir exists
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        while True:
            item = in_queue.get()
            if item is None:
                completed += 1
                if completed == n_workers:
                    # Flush remainder
                    if buffer:
                        if not header_written:
                            fieldnames = list(buffer[0].keys())
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            header_written = True
                        writer.writerows(buffer)
                        buffer.clear()
                    break
                continue

            buffer.append(item)
            if len(buffer) >= flush_every:
                if not header_written:
                    fieldnames = list(buffer[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    header_written = True
                writer.writerows(buffer)
                buffer.clear()


def _chunk(lst: List[Any], n: int) -> List[List[Any]]:
    if n <= 0:
        return [lst]
    k = max(1, math.ceil(len(lst) / n))
    return [lst[i:i + k] for i in range(0, len(lst), k)]


def main():
    parser = argparse.ArgumentParser(description='Generate strictly balanced GenAIRR dataset')
    parser.add_argument('--genairr_dataconfig', required=True, nargs='+', help='One or more built-in config names or paths to pickled DataConfigs')
    parser.add_argument('--repeats', type=int, default=1, help='Multiply allele lists uniformly to scale dataset size')
    parser.add_argument('--processes', type=int, default=max(1, cpu_count() - 1), help='Number of worker processes to spawn')
    parser.add_argument('--output_csv', required=True, help='Destination CSV file path')
    parser.add_argument('--num_sequences', '-n', type=int, default=None,
                        help='Target total sequences to generate (approximate). Each unique allele combination yields 2 sequences (productive and non-productive). When provided, the generator samples combinations according to balance_mode to get as close as possible to this number.')
    parser.add_argument('--balance_mode', choices=['category', 'config-total', 'none'], default='category',
                        help='How to balance across multiple dataconfigs: "category" (default) balances per category to the min across configs that have that category; "config-total" balances total combos per dataconfig to the min across configs; "none" keeps all combos from all configs (may skew totals).')
    parser.add_argument('--no_balance_across_configs', action='store_true', help='Deprecated: same as --balance_mode none')
    args = parser.parse_args()

    # Support multiple dataconfigs: build combined, shuffled task list tagged with chain_type
    dataconfigs_input: List[str] = args.genairr_dataconfig
    tasks: List[Tuple[str, Optional[str], Tuple[Optional[Any], Optional[Any], Optional[Any]]]] = []
    vj_target = vd_target = dj_target = vdj_target = 0
    # For balancing: hold per-dataconfig category combos
    per_dc_category_combos: List[Dict[str, List[Tuple[Optional[Any], Optional[Any], Optional[Any]]]]] = []
    per_dc_chain_type: List[Optional[str]] = []
    per_dc_all_combos: List[List[Tuple[Optional[Any], Optional[Any], Optional[Any]]]] = []
    for dc_name_or_path in dataconfigs_input:
        with _silence_stdio():
            dc = load_genairr_dataconfig(dc_name_or_path)
        chain_type = getattr(dc.metadata, 'chain_type', None)
        alleles = get_alleles(dc, repeats=args.repeats)
        v_list = alleles['V']
        j_list = alleles['J']
        d_list = alleles.get('D', None)
        combos = generate_all_allele_combinations(v_list, j_list, d_list)
        per_dc_all_combos.append(combos)
        # Partition by category for balancing
        cat_map: Dict[str, List[Tuple[Optional[Any], Optional[Any], Optional[Any]]]] = {
            'VJ': [], 'VD': [], 'DJ': [], 'VDJ': []
        }
        for (v, d, j) in combos:
            if v is not None and d is None and j is not None:
                cat_map['VJ'].append((v, d, j))
            elif v is not None and d is not None and j is None:
                cat_map['VD'].append((v, d, j))
            elif v is None and d is not None and j is not None:
                cat_map['DJ'].append((v, d, j))
            elif v is not None and d is not None and j is not None:
                cat_map['VDJ'].append((v, d, j))
        per_dc_category_combos.append(cat_map)
        per_dc_chain_type.append(chain_type)

    # Build balanced task list across dataconfigs per category if requested
    # Determine effective balance mode
    if args.no_balance_across_configs:
        balance_mode = 'none'
    else:
        balance_mode = args.balance_mode

    selected_per_dc: List[Dict[str, List[Tuple[Optional[Any], Optional[Any], Optional[Any]]]]] = []
    cat_names = ['VJ', 'VD', 'DJ', 'VDJ']
    total_counts = {'VJ': 0, 'VD': 0, 'DJ': 0, 'VDJ': 0}

    # Determine a target number of combos if num_sequences is requested
    target_combos: Optional[int] = None
    if args.num_sequences is not None and args.num_sequences > 0:
        target_combos = max(1, (args.num_sequences + 1) // 2)  # each combo yields 2 sequences

    if len(dataconfigs_input) > 1 and balance_mode == 'config-total':
        # Balance total combos equally across dataconfigs; if target specified, split across configs
        totals = [sum(len(cat_map[c]) for c in cat_names) for cat_map in per_dc_category_combos]
        num_cfg = len(per_dc_all_combos)
        if target_combos is None:
            k_total_each = min(totals) if totals else 0
            k_all = [min(k_total_each, totals[i]) for i in range(num_cfg)]
        else:
            base = target_combos // num_cfg
            rem = target_combos - base * num_cfg
            # Limit by available totals per config
            k_all = [min(base, totals[i]) for i in range(num_cfg)]
            # Distribute remainders to configs with more availability
            order = sorted(range(num_cfg), key=lambda i: totals[i] - k_all[i], reverse=True)
            for idx in order:
                if rem <= 0:
                    break
                if k_all[idx] < totals[idx]:
                    k_all[idx] += 1
                    rem -= 1

        for i in range(num_cfg):
            all_list = per_dc_all_combos[i][:]
            random.shuffle(all_list)
            chosen = all_list[:k_all[i]]
            sel_map: Dict[str, List[Tuple[Optional[Any], Optional[Any], Optional[Any]]]] = {c: [] for c in cat_names}
            for (v, d, j) in chosen:
                if v is not None and d is None and j is not None:
                    sel_map['VJ'].append((v, d, j))
                elif v is not None and d is not None and j is None:
                    sel_map['VD'].append((v, d, j))
                elif v is None and d is not None and j is not None:
                    sel_map['DJ'].append((v, d, j))
                elif v is not None and d is not None and j is not None:
                    sel_map['VDJ'].append((v, d, j))
            for c in cat_names:
                total_counts[c] += len(sel_map[c])
            selected_per_dc.append(sel_map)
        logger.info(f"Balance mode: config-total | Per-config selected combos: {k_all} | Total combos: {sum(k_all):,}")
    else:
        # Category mode (default) or none
        # Compute category mins for category mode
        cat_mins: Dict[str, int] = {c: 0 for c in cat_names}
        if len(dataconfigs_input) > 1 and balance_mode == 'category':
            for cat in cat_names:
                counts = [len(per_dc_category_combos[i][cat]) for i in range(len(per_dc_category_combos)) if len(per_dc_category_combos[i][cat]) > 0]
                cat_mins[cat] = min(counts) if counts else 0
        if balance_mode == 'none':
            # Flatten all combos and sample up to target or take all
            flat: List[Tuple[int, Tuple[Optional[Any], Optional[Any], Optional[Any]]]] = []
            for i, all_list in enumerate(per_dc_all_combos):
                for t in all_list:
                    flat.append((i, t))
            random.shuffle(flat)
            if target_combos is None:
                chosen = flat
            else:
                chosen = flat[:target_combos]
            # Build selected_per_dc from chosen
            selected_per_dc = [{c: [] for c in cat_names} for _ in range(len(per_dc_all_combos))]
            for i, (dc_idx, (v, d, j)) in enumerate(chosen):
                if v is not None and d is None and j is not None:
                    selected_per_dc[dc_idx]['VJ'].append((v, d, j))
                    total_counts['VJ'] += 1
                elif v is not None and d is not None and j is None:
                    selected_per_dc[dc_idx]['VD'].append((v, d, j))
                    total_counts['VD'] += 1
                elif v is None and d is not None and j is not None:
                    selected_per_dc[dc_idx]['DJ'].append((v, d, j))
                    total_counts['DJ'] += 1
                elif v is not None and d is not None and j is not None:
                    selected_per_dc[dc_idx]['VDJ'].append((v, d, j))
                    total_counts['VDJ'] += 1
            logger.info("Balance mode: none")
        else:
            # Category mode: allocate combos across categories proportional to category caps (balanced availability)
            caps = {}
            weights = {}
            for cat in cat_names:
                cfgs = [i for i in range(len(per_dc_category_combos)) if len(per_dc_category_combos[i][cat]) > 0]
                m_cat = len(cfgs)
                if m_cat == 0:
                    caps[cat] = 0
                    weights[cat] = 0
                else:
                    k_cap = min(len(per_dc_category_combos[i][cat]) for i in cfgs)
                    caps[cat] = m_cat * k_cap
                    weights[cat] = caps[cat]
            total_weight = sum(weights.values())
            # If no target provided, take full balanced caps; else allocate by weights
            if target_combos is None or total_weight == 0:
                goals = {cat: caps[cat] for cat in cat_names}
            else:
                # Largest remainder method
                raw = {cat: (weights[cat] * target_combos / total_weight) for cat in cat_names}
                goals = {cat: int(min(caps[cat], math.floor(raw[cat]))) for cat in cat_names}
                remainder = target_combos - sum(goals.values())
                rem_order = sorted(cat_names, key=lambda c: (raw[c] - math.floor(raw[c])), reverse=True)
                for cat in rem_order:
                    if remainder <= 0:
                        break
                    if goals[cat] < caps[cat]:
                        goals[cat] += 1
                        remainder -= 1
            # Now distribute per config equally up to availability
            selected_per_dc = [{c: [] for c in cat_names} for _ in range(len(per_dc_category_combos))]
            for cat in cat_names:
                cfgs = [i for i in range(len(per_dc_category_combos)) if len(per_dc_category_combos[i][cat]) > 0]
                m_cat = len(cfgs)
                if m_cat == 0 or goals[cat] == 0:
                    continue
                base = goals[cat] // m_cat
                rem = goals[cat] - base * m_cat
                # Shuffle within each config list and take base, then remainder round-robin by availability
                # Prepare shuffled lists
                shuffled_lists = {}
                avails = {}
                for i in cfgs:
                    local = per_dc_category_combos[i][cat][:]
                    random.shuffle(local)
                    shuffled_lists[i] = local
                    avails[i] = len(local)
                take = {i: min(base, avails[i]) for i in cfgs}
                # Distribute remainder
                # Order by remaining availability descending
                order = sorted(cfgs, key=lambda i: (avails[i] - take[i]), reverse=True)
                for i in order:
                    if rem <= 0:
                        break
                    if take[i] < avails[i]:
                        take[i] += 1
                        rem -= 1
                # Commit selections
                for i in cfgs:
                    chosen = shuffled_lists[i][:take[i]]
                    selected_per_dc[i][cat].extend(chosen)
                    total_counts[cat] += len(chosen)
            logger.info(f"Balance mode: category | Category goals (combos): { {k: goals[k] for k in cat_names} }")

    # Assemble tasks from selections and shuffle globally
    tasks.clear()
    for i, sel_map in enumerate(selected_per_dc):
        chain_type = per_dc_chain_type[i]
        for cat in cat_names:
            for (v, d, j) in sel_map[cat]:
                tasks.append((dataconfigs_input[i], chain_type, (v, d, j)))
    random.shuffle(tasks)

    # Targets are now based on selected counts across all dataconfigs
    vj_target = total_counts['VJ'] * 2
    vd_target = total_counts['VD'] * 2
    dj_target = total_counts['DJ'] * 2
    vdj_target = total_counts['VDJ'] * 2
    total_combos = len(tasks)
    total_sequences = total_combos * 2
    logger.info(f"Dataconfigs: {len(dataconfigs_input)} | Total unique allele combinations (balance_mode={balance_mode}): {total_combos:,}")
    logger.info(f"Total sequences to generate (x2 prod/nonprod): {total_sequences:,}")
    if args.num_sequences is not None:
        delta = abs(total_sequences - args.num_sequences)
        if delta > 2:
            logger.warning(f"Requested ~{args.num_sequences:,} sequences, planned {total_sequences:,}. Difference: {delta:,} (limited by availability and even pairing)")

    n_proc = min(args.processes, max(1, total_combos))
    chunks = _chunk(tasks, n_proc)
    # Free the large tasks list to reduce peak memory in parent process
    try:
        del tasks
        gc.collect()
    except Exception:
        pass
    logger.info(f"Spawning {len(chunks)} worker processes")

    q: Queue = Queue(maxsize=10000)
    writer = Process(target=_writer_process, args=(args.output_csv, q, len(chunks)))
    writer.start()

    # Shared progress structures
    # Per-worker progress (combos done), and worker totals per assigned chunk
    per_worker_progress = Array('i', [0] * len(chunks))
    worker_totals = Array('i', [len(chunk) for chunk in chunks])
    # Category sequence counters (each combo adds 2 sequences)
    vj_seq_counter = Value('i', 0)
    vd_seq_counter = Value('i', 0)
    dj_seq_counter = Value('i', 0)
    vdj_seq_counter = Value('i', 0)
    error_counter = Value('i', 0)
    # Start time for ETA
    start_time_val = Value('d', time.time())
    done_event = Event()

    monitor = Process(
        target=_monitor_process,
        args=(
            per_worker_progress,
            worker_totals,
            vj_seq_counter,
            vd_seq_counter,
            dj_seq_counter,
            vdj_seq_counter,
            error_counter,
            vj_target,
            vd_target,
            dj_target,
            vdj_target,
            total_combos,
            start_time_val,
            done_event,
        ),
    )
    monitor.start()

    workers: List[Process] = []
    for worker_id, chunk in enumerate(chunks):
        p = Process(
            target=_worker_emit_records_multi,
            args=(
                worker_id,
                chunk,
                q,
                per_worker_progress,
                vj_seq_counter,
                vd_seq_counter,
                dj_seq_counter,
                vdj_seq_counter,
                error_counter,
            ),
        )
        p.start()
        workers.append(p)

    # Wait for all workers to finish
    for p in workers:
        p.join()
    # Wait for writer to flush and exit
    writer.join()
    # Signal monitor to stop and join
    done_event.set()
    monitor.join()


if __name__ == '__main__':
    main()


