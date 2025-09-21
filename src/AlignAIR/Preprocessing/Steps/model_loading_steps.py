import pickle
from GenAIRR.dataconfig import DataConfig
from typing import cast

from AlignAIR.Data import SingleChainDataset, MultiChainDataset, MultiDataConfigContainer
from AlignAIR.Models.SingleChainAlignAIR.SingleChainAlignAIR import SingleChainAlignAIR
from AlignAIR.Models.MultiChainAlignAIR.MultiChainAlignAIR import MultiChainAlignAIR
from AlignAIR.Preprocessing.LongSequence.FastKmerDensityExtractor import FastKmerDensityExtractor
from AlignAIR.PretrainedComponents import builtin_orientation_classifier
from AlignAIR.Step.Step import Step
from AlignAIR.Utilities.step_utilities import FileInfo, MultiFileInfoContainer

BACKEND_ENGINE = 'tensorflow'

class ModelLoadingStep(Step):
    def __init__(self, name):
        super().__init__(name)


    def load_model(self, file_info,
                   dataconfig: MultiDataConfigContainer,
                   model_checkpoint: str,
                   max_sequence_size: int = 576):
        from pathlib import Path
        checkpoint_path = Path(model_checkpoint)

        # Prefer bundle directory if provided (contains config.json)
        if checkpoint_path.is_dir() and (checkpoint_path / 'config.json').exists():
            # Load via serialization bundle (contains dataconfig and structural config)
            self.log("Detected pretrained bundle; using from_pretrained()")
            # Decide single vs multi by inspecting config via load_bundle
            try:
                from AlignAIR.Serialization.io import load_bundle
                cfg, _dc, _meta = load_bundle(checkpoint_path)
                model_type = getattr(cfg, 'model_type', 'single_chain')
            except Exception:
                model_type = 'single_chain'

            if model_type == 'multi_chain':
                model = MultiChainAlignAIR.from_pretrained(checkpoint_path.as_posix())
                self.log("Using MultiChainAlignAIR model (bundle)")
            else:
                model = SingleChainAlignAIR.from_pretrained(checkpoint_path.as_posix())
                self.log("Using SingleChainAlignAIR model (bundle)")
            self.log(f"Loading: {checkpoint_path.name}")
            self.log("Model Loaded Successfully")

            # If we loaded a SavedModel wrapper, skip TF weight diagnostics (no trainable_weights)
            from AlignAIR.Serialization.saved_model_wrapper import SavedModelInferenceWrapper as _SMW
            if isinstance(model, _SMW):
                self.log("Loaded SavedModel wrapper; skipping variable-level diagnostics.")
                return model

            # Weight sanity diagnostic: compare to a freshly initialized model with same config (eager models only)
            try:
                import tensorflow as tf
                def _sum_l2(ws):
                    total = 0.0
                    for w in ws:
                        try:
                            total += float(tf.norm(w).numpy())
                        except Exception:
                            pass
                    return total

                def _sum_l2_diff(a, b):
                    total = 0.0
                    for wa, wb in zip(a, b):
                        if getattr(wa, 'shape', None) != getattr(wb, 'shape', None):
                            continue
                        try:
                            total += float(tf.norm(wa - wb).numpy())
                        except Exception:
                            pass
                    return total

                # Build a fresh model with the same architecture from bundle cfg
                if model_type == 'multi_chain':
                    rand_model = MultiChainAlignAIR(
                        max_seq_length=cfg.max_seq_length,
                        dataconfigs=_dc,
                        v_allele_latent_size=cfg.v_allele_latent_size,
                        d_allele_latent_size=cfg.d_allele_latent_size,
                        j_allele_latent_size=cfg.j_allele_latent_size,
                    )
                else:
                    rand_model = SingleChainAlignAIR(
                        max_seq_length=cfg.max_seq_length,
                        dataconfig=_dc,
                        v_allele_latent_size=cfg.v_allele_latent_size,
                        d_allele_latent_size=cfg.d_allele_latent_size,
                        j_allele_latent_size=cfg.j_allele_latent_size,
                    )
                # Build variables
                _ = rand_model({"tokenized_sequence": tf.zeros((1, cfg.max_seq_length), dtype=tf.float32)}, training=False)
                try:
                    _ = model({"tokenized_sequence": tf.zeros((1, cfg.max_seq_length), dtype=tf.float32)}, training=False)
                except Exception:
                    pass

                lw = list(model.trainable_weights)
                rw = list(rand_model.trainable_weights)
                loaded_norm = _sum_l2(lw)
                rand_norm = _sum_l2(rw)
                diff_norm = _sum_l2_diff(lw, rw)
                rel_diff_loaded = diff_norm / (loaded_norm + 1e-8)
                rel_diff_rand = diff_norm / (rand_norm + 1e-8)
                self.log(f"Weight diagnostics — loaded_norm={loaded_norm:.4f}, rand_norm={rand_norm:.4f}, "
                         f"rel_diff_loaded={rel_diff_loaded:.4f}, rel_diff_rand={rel_diff_rand:.4f}")
                # Warn if suspiciously low relative difference (indicating near-random weights)
                if rel_diff_loaded < 0.05 or rel_diff_rand < 0.05:
                    self.log("WARNING: Loaded weights are unexpectedly close to random initialization — please verify bundle.")

                # Focused diagnostics on allele classification heads
                def _layer_norms(layer):
                    norms = []
                    for var in getattr(layer, 'weights', []):
                        try:
                            norms.append(float(tf.norm(var).numpy()))
                        except Exception:
                            pass
                    return norms

                def _layer_diff_norms(layer_a, layer_b):
                    diffs = []
                    for va, vb in zip(getattr(layer_a, 'weights', []), getattr(layer_b, 'weights', [])):
                        if getattr(va, 'shape', None) != getattr(vb, 'shape', None):
                            diffs.append(float('nan'))
                            continue
                        try:
                            diffs.append(float(tf.norm(va - vb).numpy()))
                        except Exception:
                            diffs.append(float('nan'))
                    return diffs

                try:
                    heads = []
                    if hasattr(model, 'v_allele_call_head'):
                        heads.append(('v_allele', model.v_allele_call_head, rand_model.v_allele_call_head))
                    if hasattr(model, 'j_allele_call_head'):
                        heads.append(('j_allele', model.j_allele_call_head, rand_model.j_allele_call_head))
                    if hasattr(model, 'd_allele_call_head'):
                        heads.append(('d_allele', model.d_allele_call_head, rand_model.d_allele_call_head))

                    for name, l_loaded, l_rand in heads:
                        n_loaded = _layer_norms(l_loaded)
                        n_rand = _layer_norms(l_rand)
                        n_diff = _layer_diff_norms(l_loaded, l_rand)
                        self.log(f"Head '{name}' weight norms — loaded={n_loaded}, rand={n_rand}, diff_vs_rand={n_diff}")
                        # Heuristic warning: if all diffs are very small, likely not properly loaded
                        finite_diffs = [d for d in n_diff if d == d]  # filter NaNs
                        if finite_diffs:
                            avg_diff = sum(finite_diffs) / max(1, len(finite_diffs))
                            avg_loaded = sum(n_loaded) / max(1, len(n_loaded)) if n_loaded else 0.0
                            if avg_loaded > 0 and avg_diff / (avg_loaded + 1e-8) < 0.05:
                                self.log(f"WARNING: Allele head '{name}' weights appear very close to random init — check for shape/name mismatches during loading.")
                except Exception as _head_err:
                    self.log(f"Allele head diagnostics skipped due to error: {_head_err}")
            except Exception as diag_err:
                self.log(f"Weight diagnostic skipped due to error: {diag_err}")
            return model

        # Legacy path: build dataset-driven model and restore weights
        # Determine if we're in training mode (multiple files) or prediction mode (single file)
        is_training_mode = hasattr(file_info, 'file_infos') and len(file_info) > 1
        is_multi_chain = len(dataconfig) > 1

        # max_sequence_size provided for legacy checkpoint path model building

        if is_multi_chain:
            if is_training_mode:
                # Training mode: Multiple files, each for different chain types
                data_paths = [fi.path for fi in file_info]
                self.log(f"Multi-chain training mode: {len(data_paths)} files for {len(dataconfig)} configs")
            else:
                # Prediction mode: Single file with mixed sequences to classify
                if hasattr(file_info, 'file_infos'):
                    # MultiFileInfoContainer with single file
                    data_paths = [file_info[0].path]
                else:
                    # Single FileInfo
                    data_paths = [file_info.path]
                self.log(f"Multi-chain prediction mode: 1 file for {len(dataconfig)} configs")

            dataset = MultiChainDataset(
                data_paths=data_paths,
                dataconfigs=dataconfig,  # Pass MultiDataConfigContainer directly
                max_sequence_length=max_sequence_size,
                evaluation_only=True,
            )
        else:
            # Single chain mode - use existing logic
            if hasattr(file_info, 'file_infos'):
                # MultiFileInfoContainer with single file
                data_path = file_info[0].path
            else:
                # Single FileInfo
                data_path = file_info.path

            dataset = SingleChainDataset(
                data_path=data_path,
                dataconfig=dataconfig[0],  # Unwrap to single DataConfig
                max_sequence_length=max_sequence_size,
                evaluation_only=True,
                use_streaming=True,
            )

        model_params = dataset.generate_model_params()

        # Choose appropriate model class based on whether it's multi-chain or single-chain
        if is_multi_chain:
            model_params['dataconfigs'] = dataconfig  # Pass the MultiDataConfigContainer
            model = MultiChainAlignAIR(**model_params)
            self.log("Using MultiChainAlignAIR model")
        else:
            model = SingleChainAlignAIR(**model_params)
            self.log("Using SingleChainAlignAIR model")

        # Legacy weight file loading and TF checkpoint prefix compatibility (Keras 3-safe)
        import tensorflow as tf
        ckpt_path = Path(model_checkpoint)
        try:
            dummy = {"tokenized_sequence": tf.zeros((1, max_sequence_size), dtype=tf.float32)}
            _ = model(dummy, training=False)
        except Exception:
            pass

        # Case 1: direct Keras-supported file (.weights.h5, .h5, .keras)
        if ckpt_path.suffix in {'.weights.h5', '.h5', '.keras'}:
            model.load_weights(str(ckpt_path))
        # Case 2: TF checkpoint prefix (prefix with accompanying .index/.data-00000-of-00001)
        elif (ckpt_path.parent / f"{ckpt_path.name}.index").exists():
            tf_ckpt = tf.train.Checkpoint(model=model)
            status = tf_ckpt.restore(str(ckpt_path))
            if hasattr(status, 'expect_partial'):
                status.expect_partial()
        else:
            # Fallback attempt: try load_weights and let it raise a helpful error
            model.load_weights(str(ckpt_path))
        self.log(f"Loading: {model_checkpoint.split('/')[-1]}")
        self.log("Model Loaded Successfully")

        return model

    def process(self, predict_object):
        return predict_object

    def execute(self, predict_object):
        self.log("Loading main model...")
        # Prefer modern bundle path; fall back to legacy model_checkpoint if present
        model_path = getattr(predict_object.script_arguments, 'model_dir', None)
        if not model_path:
            model_path = getattr(predict_object.script_arguments, 'model_checkpoint', None)
        if not model_path:
            raise ValueError("No model path provided. Use --model_dir (bundle) or legacy --model_checkpoint.")

        predict_object.model = self.load_model(
            predict_object.file_info,
            predict_object.dataconfig,
            model_path,
            576,  # legacy default; bundles ignore this
        )
        self.log("Main Model Loaded...")

        # Ensure predict_object.dataconfig reflects the model's own config (bundle-aware)
        try:
            m = predict_object.model
            dc_container = None
            if hasattr(m, 'dataconfigs') and getattr(m, 'dataconfigs') is not None:
                dc_container = m.dataconfigs
            elif hasattr(m, 'dataconfig') and getattr(m, 'dataconfig') is not None:
                # Wrap single DataConfig for uniform downstream handling
                try:
                    single_dc = m.dataconfig
                    if single_dc is not None:
                        dc_container = MultiDataConfigContainer([single_dc])
                except Exception:
                    dc_container = None
            if dc_container is not None:
                predict_object.dataconfig = dc_container
        except Exception:
            pass

        # Load DNA orientation model if required
        if predict_object.script_arguments.fix_orientation:
            self.log("Loading Orientation Pipeline...")
            path = predict_object.script_arguments.custom_orientation_pipeline_path
            if path:
                with open(path, 'rb') as h:
                    predict_object.orientation_pipeline = pickle.load(h)
            else:
                # For multi-chain, use the first chain type for orientation
                dc_obj = getattr(predict_object, 'dataconfig', None)
                try:
                    is_multi = (dc_obj is not None) and hasattr(dc_obj, '__len__') and len(dc_obj) > 1
                except Exception:
                    is_multi = False
                if dc_obj is None:
                    chain_type_val = None
                else:
                    chain_type_val = dc_obj.chain_types()[0] if is_multi else dc_obj.metadata.chain_type
                # Map common strings to ChainType enum
                from GenAIRR.dataconfig.enums import ChainType as _CT
                if isinstance(chain_type_val, _CT):
                    chain_type_enum: _CT = chain_type_val
                else:
                    s = str(chain_type_val).lower()
                    if 'kappa' in s or s in {'igk', 'bcr_light_kappa'}:
                        chain_type_enum = _CT.BCR_LIGHT_KAPPA
                    elif 'lambda' in s or s in {'igl', 'bcr_light_lambda'}:
                        chain_type_enum = _CT.BCR_LIGHT_LAMBDA
                    elif 'tcrb' in s or 'beta' in s:
                        chain_type_enum = _CT.TCR_BETA
                    else:
                        chain_type_enum = _CT.BCR_HEAVY
                predict_object.orientation_pipeline = builtin_orientation_classifier(chain_type_enum)
            self.log('Orientation Pipeline Loaded Successfully')

        self.log("Loading Fitting Kmer Density Model...")

        ref_alleles = []
        
        # Handle both single and multi-chain dataconfigs
        dc_obj = getattr(predict_object, 'dataconfig', None)
        is_single = False
        try:
            is_single = (dc_obj is not None) and hasattr(dc_obj, '__len__') and len(dc_obj) == 1
        except Exception:
            is_single = False
        if is_single and dc_obj is not None:
            # Single chain
            ref_alleles = []
            for a in dc_obj.allele_list('v'):
                seq = getattr(a, 'ungapped_seq', '')
                if isinstance(seq, str):
                    ref_alleles.append(seq.upper())
            for a in dc_obj.allele_list('j'):
                seq = getattr(a, 'ungapped_seq', '')
                if isinstance(seq, str):
                    ref_alleles.append(seq.upper())
            if dc_obj.metadata.has_d:
                for a in dc_obj.allele_list('d'):
                    seq = getattr(a, 'ungapped_seq', '')
                    if isinstance(seq, str):
                        ref_alleles.append(seq.upper())
        else:
            # Multi-chain: aggregate alleles from all configs
            if dc_obj is None:
                ref_alleles = []
            else:
                for dc in dc_obj:
                    for a in dc.allele_list('v'):
                        seq = getattr(a, 'ungapped_seq', '')
                        if isinstance(seq, str):
                            ref_alleles.append(seq.upper())
                    for a in dc.allele_list('j'):
                        seq = getattr(a, 'ungapped_seq', '')
                        if isinstance(seq, str):
                            ref_alleles.append(seq.upper())
                    if dc.metadata.has_d:
                        for a in dc.allele_list('d'):
                            seq = getattr(a, 'ungapped_seq', '')
                            if isinstance(seq, str):
                                ref_alleles.append(seq.upper())
            
            # Remove duplicates while preserving order
            ref_alleles = list(dict.fromkeys(ref_alleles))
        # Choose max length based on loaded model to keep everything consistent
        max_len = getattr(predict_object.model, 'max_seq_length', 576)
        predict_object.candidate_sequence_extractor = FastKmerDensityExtractor(11, max_length=max_len, allowed_mismatches=0)
        predict_object.candidate_sequence_extractor.fit(ref_alleles)
        self.log("Kmer Density Model Fitted Successfully...")
        return predict_object


if BACKEND_ENGINE == 'torch':

    # Imports are placed inside methods to avoid hard dependency when not using torch backend
    from AlignAIR.Pytorch.Dataset import CSVReaderDataset
    from AlignAIR.Pytorch.HeavyChainAlignAIR import HeavyChainAlignAIR
    from AlignAIR.Pytorch.InputPreProcessors import HeavyChainInputPreProcessor
    from AlignAIR.Pytorch.Loss import AlignAIRHeavyChainLoss
    from AlignAIR.Pytorch.Trainer import AlignAIRTrainer

    class PytorchModelLoadingStep(Step):
        def __init__(self, name, logger=None):
            super().__init__(name)

        def load_model(self, sequences, chain_type, model_checkpoint, max_sequence_size, config=None):
            try:
                import torch  # type: ignore
            except Exception as _imp_err:  # pragma: no cover - optional dependency
                torch = None  # type: ignore
            if 'torch' not in globals() or torch is None:  # type: ignore
                raise ImportError("PyTorch is not installed but BACKEND_ENGINE=='torch'.")
            if chain_type == 'heavy':
                heavy_cfg = config['heavy'] if isinstance(config, dict) and 'heavy' in config else None
                pre_processor = HeavyChainInputPreProcessor(heavy_chain_dataconfig=heavy_cfg)
                dataset = CSVReaderDataset(
                    csv_file=sequences,
                    preprocessor=pre_processor,  # Custom preprocessing logic
                    batch_size=64,
                    separator=','
                )

            elif chain_type == 'light':
                # TO DO
                pass
            else:
                raise ValueError(f'Unknown Chain Type: {chain_type}')

            v_allele_count = pre_processor.v_allele_count
            d_allele_count = pre_processor.d_allele_count
            j_allele_count = pre_processor.j_allele_count

            # 3. Initialize Model
            model = HeavyChainAlignAIR(
                max_seq_length=576,
                v_allele_count=v_allele_count,
                d_allele_count=d_allele_count,
                j_allele_count=j_allele_count
            )

            # 4. Define Loss Function
            loss_function = AlignAIRHeavyChainLoss

            # 5. Instantiate Trainer
            trainer = AlignAIRTrainer(
                model=model,
                dataset=dataset,
                loss_function=loss_function,
                optimizer=torch.optim.Adam,
                optimizer_params={"lr": 1e-3},
                epochs=10,
                batch_size=64,
                log_to_file=True,
                log_file_path="C:/Users/tomas/Downloads/",
                verbose=2,
                batches_per_epoch=64
            )

            MODEL_CHECKPOINT = model_checkpoint

            trainer.load_model(MODEL_CHECKPOINT)

            self.log(f"Loading: {MODEL_CHECKPOINT.split('/')[-1]}")
            self.log(f"Model Loaded Successfully")
            return trainer.model

        def process(self, predict_object):
            return predict_object

        def execute(self, predict_object):
            self.log("Loading main model...")
            predict_object.model = self.load_model(
                predict_object.script_arguments.sequences,
                predict_object.script_arguments.chain_type,
                predict_object.script_arguments.model_checkpoint,
                predict_object.script_arguments.max_input_size,
                predict_object.data_config
            )
            self.log("Main Model Loaded...")

            # Load DNA orientation model if required
            if predict_object.script_arguments.fix_orientation:
                self.log("Loading Orientation Pipeline...")
                path = predict_object.script_arguments.custom_orientation_pipeline_path
                if path:
                    with open(path, 'rb') as h:
                        predict_object.orientation_pipeline = pickle.load(h)
                else:
                    predict_object.orientation_pipeline = builtin_orientation_classifier()
                self.log('Orientation Pipeline Loaded Successfully')

            self.log("Loading Fitting Kmer Density Model...")

            ref_alleles = None
            if predict_object.chain_type == 'heavy':
                data_config = predict_object.data_config[predict_object.chain_type]
                hc_alleles = [i.ungapped_seq.upper() for j in data_config.v_alleles for i in data_config.v_alleles[j]]
                hc_alleles += [i.ungapped_seq.upper() for j in data_config.j_alleles for i in data_config.j_alleles[j]]
                hc_alleles += [i.ungapped_seq.upper() for j in data_config.d_alleles for i in data_config.d_alleles[j]]
                ref_alleles = hc_alleles
            elif predict_object.chain_type == 'light':
                data_config_lambda = predict_object.data_config['lambda']
                data_config_kappa = predict_object.data_config['kappa']

                lc_alleles = [i.ungapped_seq.upper() for j in data_config_lambda.v_alleles for i in
                              data_config_lambda.v_alleles[j]] + \
                             [i.ungapped_seq.upper() for j in data_config_lambda.v_alleles for i in
                              data_config_lambda.v_alleles[j]]
                lc_alleles += [i.ungapped_seq.upper() for j in data_config_kappa.j_alleles for i in
                               data_config_kappa.j_alleles[j]] + \
                              [i.ungapped_seq.upper() for j in data_config_kappa.j_alleles for i in
                               data_config_kappa.j_alleles[j]]
                ref_alleles = lc_alleles

            predict_object.candidate_sequence_extractor = FastKmerDensityExtractor(11, max_length=576, allowed_mismatches=0)
            predict_object.candidate_sequence_extractor.fit(ref_alleles)
            self.log("Kmer Density Model Fitted Successfully...")
            return predict_object
