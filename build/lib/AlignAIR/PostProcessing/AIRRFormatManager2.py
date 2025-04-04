import pandas as pd
from multiprocessing import Pool
from GenAIRR.utilities import translate


class AIRRFormatManagerOptimized:
    def __init__(self, predict_object):
        self.po = predict_object
        self.chain = predict_object.script_arguments.chain_type
        self.data_config = predict_object.data_config_library.data_configs
        self.reference_map, self.j_anchor_dict = self._derive_allele_dictionaries()

    def _derive_allele_dictionaries(self):
        return self._derive_heavy_alleles() if self.chain == 'heavy' else self._derive_light_alleles()

    def _derive_light_alleles(self):
        def _dict(alleles, attr): return self._build_allele_dict(alleles, attr)
        return {
            'v': {**_dict(self.data_config["kappa"].v_alleles, "gapped_seq"),
                  **_dict(self.data_config["lambda"].v_alleles, "gapped_seq")},
            'j': {**_dict(self.data_config["kappa"].j_alleles, "ungapped_seq"),
                  **_dict(self.data_config["lambda"].j_alleles, "ungapped_seq")}
        }, {**_dict(self.data_config["kappa"].j_alleles, "anchor"),
            **_dict(self.data_config["lambda"].j_alleles, "anchor")}

    def _derive_heavy_alleles(self):
        def _dict(alleles, attr): return self._build_allele_dict(alleles, attr)
        return {
            'v': _dict(self.data_config["heavy"].v_alleles, "gapped_seq"),
            'd': _dict(self.data_config["heavy"].d_alleles, "ungapped_seq"),
            'j': _dict(self.data_config["heavy"].j_alleles, "ungapped_seq")
        }, _dict(self.data_config["heavy"].j_alleles, "anchor")

    def _build_allele_dict(self, allele_set, attr):
        return {allele.name: getattr(allele, attr).upper() if "seq" in attr else getattr(allele, attr)
                for group in allele_set.values() for allele in group}

    def _get_reference_seq(self, call, call_type, start, end=None):
        ref_seq = self.reference_map.get(call_type, {}).get(call, "")
        if not ref_seq:
            return ""
        return ref_seq[:start] if end is None else ref_seq[start:end]

    def _sequence_alignment_worker(self, row):
        if row['skip_processing']:
            return pd.NA
        v_call = row['v_call'].split(',')[0]
        v_ref = self.reference_map['v'].get(v_call, '')
        if not v_ref:
            return ''
        v_ref = v_ref[:row['v_germline_end']]
        seq = row['sequence'][row['v_sequence_start']:row['j_sequence_end']]
        if row['v_germline_start'] > 0:
            seq = '.' * row['v_germline_start'] + seq
        seq_iter = iter(seq)
        aligned_seq = []
        started = False
        for ref_base in v_ref:
            if ref_base != '.':
                started = True
                aligned_seq.append(next(seq_iter, '.'))
            else:
                aligned_seq.append('.' if started else next(seq_iter, '.'))
        aligned_seq.extend(seq_iter)
        return ''.join(aligned_seq)

    def _germline_alignment_worker(self, row):
        if row['skip_processing']:
            return pd.NA
        v_call = row['v_call'].split(',')[0]
        j_call = row['j_call'].split(',')[0]
        v_ref = self._get_reference_seq(v_call, 'v', 0, row['v_germline_end'])
        j_ref = self._get_reference_seq(j_call, 'j', row['j_germline_start'], row['j_germline_end'])

        if self.chain == 'heavy':
            d_call = row['d_call'].split(',')[0]
            if d_call == 'Short-D':
                d_region = row['sequence'][row['v_sequence_end']:row['j_sequence_start']]
            else:
                d_ref = self._get_reference_seq(d_call, 'd', row['d_germline_start'], row['d_germline_end'])
                np1 = row['sequence'][row['v_sequence_end']:row['d_sequence_start']]
                np2 = row['sequence'][row['d_sequence_end']:row['j_sequence_start']]
                d_region = np1 + d_ref + np2
            return v_ref + d_region + j_ref
        else:
            np1 = row['sequence'][row['v_sequence_end']:row['j_sequence_start']]
            return v_ref + np1 + j_ref

    def _run_parallel(self, func, rows, n_processes=8):
        with Pool(n_processes) as pool:
            return pool.map(func, rows)

    def _translate_column(self, series):
        return series.dropna().map(translate).reindex(series.index, fill_value=pd.NA)

    def build_dataframe(self):
        po = self.po
        n = len(po.sequences)
        df = pd.DataFrame({
            'sequence_id': [f'Query_{i + 1}' for i in range(n)],
            'sequence': po.sequences,
            'productive': po.processed_predictions['productive'],
            'v_call': [','.join(i) for i in po.selected_allele_calls['v']],
            'j_call': [','.join(i) for i in po.selected_allele_calls['j']],
            'v_sequence_start': [i['start_in_seq'] for i in po.germline_alignments['v']],
            'v_sequence_end': [i['end_in_seq'] for i in po.germline_alignments['v']],
            'v_germline_start': [max(0, i['start_in_ref']) for i in po.germline_alignments['v']],
            'v_germline_end': [i['end_in_ref'] for i in po.germline_alignments['v']],
            'j_sequence_start': [i['start_in_seq'] for i in po.germline_alignments['j']],
            'j_sequence_end': [i['end_in_seq'] for i in po.germline_alignments['j']],
            'j_germline_start': [max(0, i['start_in_ref']) for i in po.germline_alignments['j']],
            'j_germline_end': [i['end_in_ref'] for i in po.germline_alignments['j']],
            'mutation_rate': po.processed_predictions.get('mutation_rate', [pd.NA] * n),
            'ar_indels': po.processed_predictions.get('indel_count', [pd.NA] * n),
            'v_likelihoods': po.likelihoods_of_selected_alleles.get('v', [pd.NA] * n),
            'j_likelihoods': po.likelihoods_of_selected_alleles.get('j', [pd.NA] * n),
            'd_likelihoods': po.likelihoods_of_selected_alleles.get('d', [pd.NA] * n) if self.chain == 'heavy' else [pd.NA] * n,
        })

        if self.chain == 'heavy':
            df['d_sequence_start'] = [i['start_in_seq'] for i in po.germline_alignments['d']]
            df['d_sequence_end'] = [i['end_in_seq'] for i in po.germline_alignments['d']]
            df['d_germline_start'] = [abs(i['start_in_ref']) for i in po.germline_alignments['d']]
            df['d_germline_end'] = [i['end_in_ref'] for i in po.germline_alignments['d']]
            df['d_call'] = [','.join(i) for i in po.selected_allele_calls['d']]
            df['locus'] = ['IGH'] * n
        else:
            df['d_sequence_start'] = pd.NA
            df['d_sequence_end'] = pd.NA
            df['d_germline_start'] = pd.NA
            df['d_germline_end'] = pd.NA
            df['d_call'] = [''] * n
            df['locus'] = ['IGK' if i == 1 else 'IGL' for i in po['type_'].astype(int).squeeze()]

        df['skip_processing'] = (df['productive'] == False) & (df['ar_indels'] > 1)

        rows = df.to_dict("records")
        df['sequence_alignment'] = self._run_parallel(self._sequence_alignment_worker, rows)
        df['germline_alignment'] = self._run_parallel(self._germline_alignment_worker, rows)
        df['sequence_alignment_aa'] = self._translate_column(df['sequence_alignment'])
        df['germline_alignment_aa'] = self._translate_column(df['germline_alignment'])

        df['v_alignment_start'] = 1
        df['v_alignment_end'] = df['v_germline_end'] - df['v_germline_start']
        df['j_alignment_start'] = df['j_germline_start']
        df['j_alignment_end'] = df['j_germline_end']
        df['d_alignment_start'] = df['d_germline_start']
        df['d_alignment_end'] = df['d_germline_end']

        def slice_or_na(seq, s, e):
            try:
                return seq[s:e] if pd.notna(s) and pd.notna(e) else pd.NA
            except Exception:
                return pd.NA

        df['v_sequence_alignment'] = [slice_or_na(a, s, e) for a, s, e in zip(df['sequence_alignment'], df['v_alignment_start'], df['v_alignment_end'])]
        df['v_germline_alignment'] = [slice_or_na(a, s, e) for a, s, e in zip(df['germline_alignment'], df['v_alignment_start'], df['v_alignment_end'])]
        df['j_sequence_alignment'] = [slice_or_na(a, s, e) for a, s, e in zip(df['sequence_alignment'], df['j_alignment_start'], df['j_alignment_end'])]
        df['j_germline_alignment'] = [slice_or_na(a, s, e) for a, s, e in zip(df['germline_alignment'], df['j_alignment_start'], df['j_alignment_end'])]

        df['v_sequence_alignment_aa'] = df['v_sequence_alignment'].map(translate)
        df['v_germline_alignment_aa'] = df['v_germline_alignment'].map(translate)
        df['j_sequence_alignment_aa'] = df['j_sequence_alignment'].map(translate)
        df['j_germline_alignment_aa'] = df['j_germline_alignment'].map(translate)

        # ---- Junction and CDR3 ----
        df['junction'] = [s[v_end:j_start] for s, v_end, j_start in zip(df['sequence'], df['v_sequence_end'], df['j_sequence_start'])]
        df['junction_length'] = df['junction'].map(lambda x: len(x) if pd.notna(x) else pd.NA)
        df['junction_aa'] = df['junction'].map(translate)
        df['junction_aa_length'] = df['junction_aa'].map(lambda x: len(x) if pd.notna(x) else pd.NA)

        # ---- N/P regions ----
        df['np1'] = [s[v_end:d_start] if pd.notna(d_start) else pd.NA for s, v_end, d_start in zip(df['sequence'], df['v_sequence_end'], df['d_sequence_start'])]
        df['np2'] = [s[d_end:j_start] if pd.notna(d_end) else pd.NA for s, d_end, j_start in zip(df['sequence'], df['d_sequence_end'], df['j_sequence_start'])]
        df['np1_length'] = df['np1'].map(lambda x: len(x) if pd.notna(x) else pd.NA)
        df['np2_length'] = df['np2'].map(lambda x: len(x) if pd.notna(x) else pd.NA)

        # ---- Framework & CDR region slicing ----
        for region, (start, end) in {
            'fwr1': (0, 78), 'cdr1': (78, 114), 'fwr2': (114, 165), 'cdr2': (165, 195),
            'fwr3': (195, 312), 'cdr3': (312, None), 'fwr4': (None, None)
        }.items():
            df[region] = [s[start:end] if pd.notna(s) else pd.NA for s in df['sequence_alignment_aa']]
            df[f"{region}_aa"] = df[region]
            df[f"{region}_start"] = start
            df[f"{region}_end"] = end

        # ---- CDR3 positions ----
        df['cdr3_start'] = df['v_sequence_end']
        df['cdr3_end'] = df['j_sequence_start']

        # ---- In-frame and stop codons ----
        df['stop_codon'] = df['sequence_alignment_aa'].map(lambda x: '*' in x if pd.notna(x) else pd.NA)
        df['vj_in_frame'] = [(cdr3_end - v_start) % 3 == 0 if pd.notna(cdr3_end) and pd.notna(v_start) else pd.NA
                             for cdr3_end, v_start in zip(df['cdr3_end'], df['v_alignment_start'])]

        return df
