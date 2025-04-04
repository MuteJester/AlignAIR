import pandas as pd
from GenAIRR.utilities import translate

class AIRRFormatManager:
    """
    Manager class for constructing an AIRR-compliant DataFrame from a prediction object.
    Handles sequence alignment, productivity flags, segment extractions,
    AIRR schema formatting, and R-compatible output.
    """

    def __init__(self, predict_object):
        self.predict_object = predict_object
        self.chain = predict_object.script_arguments.chain_type
        self.data_config = predict_object.data_config_library.data_configs
        self.reference_map, self.j_anchor_dict = self._derive_allele_dictionaries()

        self.required_columns = [
            'sequence_id', 'sequence', 'locus', 'stop_codon', 'vj_in_frame', 'productive',
            'v_call', 'd_call', 'j_call', 'sequence_alignment', 'germline_alignment',
            'sequence_alignment_aa', 'germline_alignment_aa',
            'v_sequence_alignment', 'v_sequence_alignment_aa', 'v_germline_alignment',
            'v_germline_alignment_aa', 'd_sequence_alignment', 'd_sequence_alignment_aa',
            'd_germline_alignment', 'd_germline_alignment_aa', 'j_sequence_alignment',
            'j_sequence_alignment_aa', 'j_germline_alignment', 'j_germline_alignment_aa',
            'fwr1', 'frw1_aa', 'cdr1', 'cdr1_aa', 'fwr2', 'fwr2_aa', 'cdr2', 'cdr2_aa',
            'fwr3', 'fwr3_aa', 'fwr4', 'fwr4_aa', 'cdr3', 'cdr3_aa', 'junction', 'junction_length',
            'junction_aa', 'junction_aa_length', 'v_sequence_start', 'v_sequence_end',
            'v_germline_start', 'v_germline_end', 'v_alignment_start', 'v_alignment_end', 
            'd_sequence_start', 'd_sequence_end', 'd_germline_start', 'd_germline_end', 
            'd_alignment_start', 'd_alignment_end','j_sequence_start', 'j_sequence_end',
            'j_germline_start', 'j_germline_end', 'j_alignment_start', 'j_alignment_end', 
            'frw1_start', 'frw1_end', 'cdr1_start', 'cdr1_end', 'frw2_start', 'frw2_end', 
            'cdr2_start', 'cdr2_end', 'frw3_start', 'frw3_end', 'frw4_start', 'frw4_end', 
            'cdr3_start', 'cdr3_end', 'np1', 'np1_length', 'np2', 'np2_length'
        ]

        self.extra_columns = ['v_likelihoods', 'd_likelihoods', 'j_likelihoods', 'mutation_rate', 'ar_indels']

        self.regions = {
            'fwr1': [0, 78], 'cdr1': [78, 114], 'fwr2': [114, 165], 'cdr2': [165, 195],
            'fwr3': [195, 312], 'cdr3': [312, None], 'fwr4': [None, None], 'junction': [309, None]
        }

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

    def _get_reference_seq(self, row, call_type, start_key, end_key=None):
        call = row.get(f"{call_type}_call", "").split(",")[0]
        ref_seq = self.reference_map.get(call_type, {}).get(call, "")
        if not ref_seq:
            return ""
        return ref_seq[:row[start_key]] if end_key is None else ref_seq[row[start_key]:row[end_key]]

    def get_sequence_alignment(self, row):
        """
        Gapps the aligned sequence with the V germline sequence. 
        Adapted from William Lees https://github.com/williamdlees/receptor_utils
        """
        if row.get('skip_processing', False):
            return pd.NA
        v_call = row.get('v_call', '').split(',')[0]
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

    def get_germline_alignment(self, row):
        if row.get('skip_processing', False):
            return pd.NA
        v_ref = self._get_reference_seq(row, 'v', 'v_germline_end')
        j_ref = self._get_reference_seq(row, 'j', 'j_germline_start', 'j_germline_end')
        if self.chain == 'heavy':
            d_call = row.get('d_call', '').split(',')[0]
            if d_call == 'Short-D':
                d_region = row['sequence'][row['v_sequence_end']:row['j_sequence_start']]
            else:
                d_ref = self._get_reference_seq(row, 'd', 'd_germline_start', 'd_germline_end')
                np1 = row['sequence'][row['v_sequence_end']:row['d_sequence_start']]
                np2 = row['sequence'][row['d_sequence_end']:row['j_sequence_start']]
                d_region = np1 + d_ref + np2
            return v_ref + d_region + j_ref
        else:
            np1 = row['sequence'][row['v_sequence_end']:row['j_sequence_start']]
            return v_ref + np1 + j_ref

    def _translate_alignments(self, df):
        df['sequence_alignment_aa'] = df.apply(
            lambda r: translate(r['sequence_alignment']) if pd.notna(r['sequence_alignment']) else pd.NA, axis=1)
        df['germline_alignment_aa'] = df.apply(
            lambda r: translate(r['germline_alignment']) if pd.notna(r['germline_alignment']) else pd.NA, axis=1)
        return df
    
    def _map_segment_alignment_positions(self, df):
        def compute_positions(row):
            if row.get('skip_processing', False) or pd.isna(row['sequence_alignment']):
                return pd.Series({
                    'v_alignment_start': pd.NA,
                    'v_alignment_end': pd.NA,
                    'd_alignment_start': pd.NA,
                    'd_alignment_end': pd.NA,
                    'j_alignment_start': pd.NA,
                    'j_alignment_end': pd.NA,
                })

            num_gaps = row['sequence_alignment'].count('.')
            v_start = 0
            v_end = row['v_germline_end'] + num_gaps
            
            if self.chain == 'heavy' and pd.notna(row['d_sequence_start']):
                d_start = row['d_sequence_start'] + num_gaps
                d_end = row['d_sequence_end'] + num_gaps
            else:
                d_start = pd.NA
                d_end = pd.NA

            j_start = row['j_sequence_start'] + num_gaps
            j_end = row['j_sequence_end'] + num_gaps

            return pd.Series({
                'v_alignment_start': v_start,
                'v_alignment_end': v_end,
                'd_alignment_start': d_start,
                'd_alignment_end': d_end,
                'j_alignment_start': j_start,
                'j_alignment_end': j_end
            })

        df = df.join(df.apply(compute_positions, axis=1))
        return df

    def _populate_segment_alignment_columns(self, df):
        segments = ['v', 'j'] + (['d'] if self.chain == 'heavy' else [])
        for seg in segments:
            s_col, e_col = f'{seg}_alignment_start', f'{seg}_alignment_end'
            df[f'{seg}_sequence_alignment'] = df.apply(
                lambda r: r['sequence_alignment'][r[s_col]:r[e_col]]
                if not r.get('skip_processing', False) and pd.notna(r[s_col]) and pd.notna(r[e_col]) else pd.NA, axis=1)
            df[f'{seg}_sequence_alignment_aa'] = df.apply(
                lambda r: r['sequence_alignment_aa'][r[s_col]//3:r[e_col]//3]
                if not r.get('skip_processing', False) and pd.notna(r[s_col]) and pd.notna(r[e_col]) else pd.NA, axis=1)
            df[f'{seg}_germline_alignment'] = df.apply(
                lambda r: r['germline_alignment'][r[s_col]:r[e_col]]
                if not r.get('skip_processing', False) and pd.notna(r[s_col]) and pd.notna(r[e_col]) else pd.NA, axis=1)
            df[f'{seg}_germline_alignment_aa'] = df.apply(
                lambda r: r['germline_alignment_aa'][r[s_col]//3:r[e_col]//3]
                if not r.get('skip_processing', False) and pd.notna(r[s_col]) and pd.notna(r[e_col]) else pd.NA, axis=1)
        return df

    def _add_region_columns(self, df):
        for region, (start, end) in self.regions.items():
            if region in ['cdr3', 'junction', 'fwr4']:
                continue
            df[region] = df['sequence_alignment'].str.slice(start, end)
            df[f'{region}_start'] = start
            df[f'{region}_end'] = end
            df[f'{region}_aa'] = df['sequence_alignment_aa'].str.slice(
                None if start is None else start // 3,
                None if end is None else end // 3
            )
        return df

    def _add_cdr3_and_junction_columns(self, df):
        def compute(row):
            if row.get('skip_processing', False): return pd.Series({col: pd.NA for col in [
                'junction', 'junction_aa', 'junction_length', 'junction_aa_length', 'cdr3', 'cdr3_aa', 'fwr4', 'fwr4_aa']})
            j_call = row.get('j_call', '').split(',')[0]
            j_anchor = self.j_anchor_dict.get(j_call, 0)
            j_seq_start = row['j_sequence_start'] - row['v_sequence_start'] + row['sequence_alignment'].count('.')
            junc_end = j_seq_start + j_anchor - row['j_germline_start'] + 3
            junction_nt = row['sequence_alignment'][self.regions['junction'][0]:junc_end]
            junction_aa = row['sequence_alignment_aa'][self.regions['junction'][0] // 3: junc_end // 3]
            cdr3_aa = junction_aa[1:-1] if len(junction_aa) > 2 else ''
            cdr3_start = self.regions['junction'][0] + 3
            cdr3_end = junc_end - 3
            cdr3_nt = row['sequence_alignment'][cdr3_start : cdr3_end]
            fwr4_start = cdr3_end + 1
            fwr4_end = row['j_alignment_end']
            fwr4_nt = row['sequence_alignment'][fwr4_start:fwr4_end]
            fwr4_aa = row['sequence_alignment_aa'][fwr4_start // 3: fwr4_end // 3]
            return pd.Series({
                'junction': junction_nt, 'junction_aa': junction_aa,
                'junction_length': len(junction_nt), 'junction_aa_length': len(junction_aa),
                'cdr3': cdr3_nt, 'cdr3_aa': cdr3_aa, 'fwr4': fwr4_nt, 'fwr4_aa': fwr4_aa,
                'cdr3_start': cdr3_start, 'cdr3_end': cdr3_end,
                'frw4_start': fwr4_start, 'frw4_end': fwr4_end
            })
        return df.join(df.apply(compute, axis=1))

    def _add_np_regions(self, df):
        if self.chain == 'heavy':
            df['np1'] = df.apply(lambda row: row['sequence'][row['v_sequence_end']:row['d_sequence_start']]
                                 if not row.get('skip_processing', False) else pd.NA, axis=1)
            df['np2'] = df.apply(lambda row: row['sequence'][row['d_sequence_end']:row['j_sequence_start']]
                                 if not row.get('skip_processing', False) else pd.NA, axis=1)
        else:
            df['np1'] = df.apply(lambda row: row['sequence'][row['v_sequence_end']:row['j_sequence_start']]
                                 if not row.get('skip_processing', False) else pd.NA, axis=1)
            df['np2'] = pd.NA
        df['np1_length'] = df['np1'].apply(lambda x: len(x) if pd.notna(x) else pd.NA)
        df['np2_length'] = df['np2'].apply(lambda x: len(x) if pd.notna(x) else pd.NA)
        return df

    def _add_productivity_flags(self, df):
        df['stop_codon'] = df['sequence_alignment_aa'].str.contains('\\*')
        v_start = df['v_alignment_start']
        cdr3_end = df['cdr3_end']
        cdr3_start = df['cdr3_start']
        from_j_to_start = (cdr3_end - v_start) % 3 == 0
        cdr3_len_frame = (cdr3_end - cdr3_start) % 3 == 0
        from_v_to_start = (cdr3_end - v_start) % 3 == 0
        df['vj_in_frame'] = from_j_to_start & cdr3_len_frame & from_v_to_start & (~df['stop_codon'])
        return df

    def _reorder_and_finalize_columns(self, df):
        for col in self.required_columns + self.extra_columns:
            if col not in df.columns:
                df[col] = pd.NA
        # Booleans to \"T\"/\"F\"
        for col in df.columns:
            if df[col].dtype == bool or df[col].dropna().isin([True, False]).all():
                df[col] = df[col].map({True: 'T', False: 'F'})
        # 1-based indexing
        for col in df.columns:
            if col.endswith('_start') or col.endswith('_end'):
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].apply(lambda x: x+1 if pd.notna(x) else x)
        if 'skip_processing' in df.columns:
            df.drop(columns='skip_processing', inplace=True)
        return df[self.required_columns + self.extra_columns]

    def build_dataframe(self):
        """
        Main pipeline to generate a full AIRR-compliant DataFrame.
        Applies light processing per sequence if necessary.
        """
        po = self.predict_object
        
        df = pd.DataFrame({
            'sequence_id': [f'Query_{i+1}' for i in range(len(po.sequences))],
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
            'mutation_rate': po.processed_predictions.get('mutation_rate', [pd.NA]*len(po.sequences)),
            'ar_indels': po.processed_predictions.get('indel_count', [pd.NA]*len(po.sequences)),
            'v_likelihoods': po.likelihoods_of_selected_alleles.get('v', [pd.NA]*len(po.sequences)),
            'j_likelihoods': po.likelihoods_of_selected_alleles.get('j', [pd.NA]*len(po.sequences)),
            'd_likelihoods': po.likelihoods_of_selected_alleles.get('d', [pd.NA]*len(po.sequences)) if self.chain == 'heavy' else pd.NA,
        })
        
        if self.chain == 'heavy':
            df['d_sequence_start'] = [i['start_in_seq'] for i in po.germline_alignments['d']]
            df['d_sequence_end'] = [i['end_in_seq'] for i in po.germline_alignments['d']]
            df['d_germline_start'] = [abs(i['start_in_ref']) for i in po.germline_alignments['d']]
            df['d_germline_end'] = [i['end_in_ref'] for i in po.germline_alignments['d']]
            df['d_call'] = [','.join(i) for i in po.selected_allele_calls['d']]
            df['locus'] = 'IGH'
        else:
            df['d_sequence_start'] = pd.NA
            df['d_sequence_end'] = pd.NA
            df['d_germline_start'] = pd.NA
            df['d_germline_end'] = pd.NA
            df['d_call'] = ''
            df['locus'] = ['IGK' if i == 1 else 'IGL' for i in po['type_'].astype(int).squeeze()]
        
        df['skip_processing'] = (df['productive'] == False) & (df['ar_indels'] > 1)
        df['sequence_alignment'] = df.apply(lambda r: self.get_sequence_alignment(r) if not r['skip_processing'] else pd.NA, axis=1)
        df['germline_alignment'] = df.apply(lambda r: self.get_germline_alignment(r) if not r['skip_processing'] else pd.NA, axis=1)
        
        df = self._translate_alignments(df)
        df = self._map_segment_alignment_positions(df)
        df = self._populate_segment_alignment_columns(df)
        df = self._add_region_columns(df)
        df = self._add_cdr3_and_junction_columns(df)
        df = self._add_np_regions(df)
        df = self._add_productivity_flags(df)
        df = self._reorder_and_finalize_columns(df)
        return df
