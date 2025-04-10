import re
import pandas as pd
import numpy as np
from Bio import SeqIO
from GenAIRR.utilities import translate


class AIRRFormatManager:
    def __init__(self, predict_object):
        self.predict_object = predict_object
        self.chain = predict_object.script_arguments.chain_type
        self.data_config = predict_object.data_config_library.data_configs
        self.reference_map, self.j_anchor_dict = self._derive_allele_dictionaries()

        self.regions = {
            'fwr1': [0, 78], 'cdr1': [78, 114], 'fwr2': [114, 165], 'cdr2': [165, 195],
            'fwr3': [195, 312], 'cdr3': [312, None], 'fwr4': [None, None], 'junction': [309, None]
        }

    def _derive_allele_dictionaries(self):
        return self._derive_heavy_alleles() if self.chain == 'heavy' else self._derive_light_alleles()

    def _derive_light_alleles(self):
        def _dict(alleles, attr):
            return self._build_allele_dict(alleles, attr)
        return {
            'v': {**_dict(self.data_config["kappa"].v_alleles, "gapped_seq"),
                  **_dict(self.data_config["lambda"].v_alleles, "gapped_seq")},
            'j': {**_dict(self.data_config["kappa"].j_alleles, "ungapped_seq"),
                  **_dict(self.data_config["lambda"].j_alleles, "ungapped_seq")}
        }, {
            **_dict(self.data_config["kappa"].j_alleles, "anchor"),
            **_dict(self.data_config["lambda"].j_alleles, "anchor")
        }

    def _derive_heavy_alleles(self):
        def _dict(alleles, attr):
            return self._build_allele_dict(alleles, attr)
        return {
            'v': _dict(self.data_config["heavy"].v_alleles, "gapped_seq"),
            'd': _dict(self.data_config["heavy"].d_alleles, "ungapped_seq"),
            'j': _dict(self.data_config["heavy"].j_alleles, "ungapped_seq")
        }, _dict(self.data_config["heavy"].j_alleles, "anchor")

    def _build_allele_dict(self, allele_set, attr):
        return {
            allele.name: getattr(allele, attr).upper() if "seq" in attr else getattr(allele, attr)
            for group in allele_set.values()
            for allele in group
        }

    def _get_reference_seq(self, row, call_type, start_key, end_key=None):
        call = row.get(f"{call_type}_call", "").split(",")[0]
        ref_seq = self.reference_map.get(call_type, {}).get(call, "")
        if not ref_seq:
            return ""
        return ref_seq[:row[start_key]] if end_key is None else ref_seq[row[start_key]:row[end_key]]
    
    def _extract_sequence_id_from_csv(self, file_path):
        sep = ',' if '.csv' in file_path else '\t'
        df = pd.read_csv(file_path, usecols=['sequence'], sep=sep)
        if 'sequence_id' in df.columns:
            return df['sequence_id'].tolist()
        else:
            return [f'Query_{i+1}' for i in range(len(df))]
    
    def _extract_sequence_id_from_fasta(self, file_path):
        sequence_ids = []
        for record in SeqIO.parse(file_path, "fasta"):
            sequence_ids.append(record.id)
        return sequence_ids
    
    
    def _extract_sequence_ids(self, file_path, file_type='fasta'):
        if file_type == 'csv' or file_type == 'tsv':
            return self._extract_sequence_id_from_csv(file_path)
        elif file_type == 'fasta':
            return self._extract_sequence_id_from_fasta(file_path)
        else:
            raise ValueError("Unsupported file type. Supported types are 'csv', 'tsv', and 'fasta'.")

    
    def _parse_sequence_id(self, sequence_ids):
        metadata_data = {'sequence_id': []}
        metadata_pattern = r'(\w+)=([\w,._-]+)'

        for seq_id in sequence_ids:
            metadata_strs = seq_id.split('|')
            metadata_dict = {'sequence_id': metadata_strs[0]}

            if len(metadata_strs) > 1:
                metadata_string = '|'.join(metadata_strs[1:])  # Join the rest into one string
                metadata_pairs = re.findall(metadata_pattern, metadata_string)
                for key, value in metadata_pairs:
                    ## to keep with airr format, change dupcount to duplicate_count and conscount to consensus_count
                    if key.lower() == 'conscount':
                        key = 'consensus_count'
                    elif key.lower() == 'dupcount':
                        key = 'duplicate_count'
                        
                    metadata_dict[key.lower()] = value

            for key, value in metadata_dict.items():
                if key not in metadata_data:
                    metadata_data[key] = []
                metadata_data[key].append(value)

        return metadata_data
    
    def _get_sequence_alignment(self, row):
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

    def _get_germline_alignment(self, row):
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
    
    
    def _get_alignments(self, airr_dict, n):
        
        airr_dict['skip_processing'] = [(p is False and (a or 0) > 1) for p, a in zip(airr_dict['productive'], airr_dict['ar_indels'])]

        airr_dict['sequence_alignment'] = [
            self._get_sequence_alignment({k: airr_dict[k][i] for k in airr_dict}) if not airr_dict['skip_processing'][i] else None
            for i in range(n)
        ]

        airr_dict['germline_alignment'] = [
            self._get_germline_alignment({k: airr_dict[k][i] for k in airr_dict}) if not airr_dict['skip_processing'][i] else None
            for i in range(n)
        ]
        
        return airr_dict
    
    
    def _translate_alignments(self, airr_dict):
        
        airr_dict['sequence_alignment_aa'] = [
            translate(seq) if seq is not None else None
            for seq in airr_dict['sequence_alignment']
        ]
        
        airr_dict['germline_alignment_aa'] = [
            translate(seq) if seq is not None else None
            for seq in airr_dict['germline_alignment']
        ]
        
        return airr_dict
    
    
    def _map_segment_alignment_positions(self, airr_dict):
        n = len(airr_dict['sequence_alignment'])

        v_end = [None] * n
        d_start = [None] * n
        d_end = [None] * n
        j_start = [None] * n
        j_end = [None] * n
        v_start = [0] * n

        seqs = np.array(airr_dict['sequence_alignment'], dtype=object)
        skip = np.array(airr_dict['skip_processing'])
        gaps = np.char.count(seqs.astype(str), '.')

        for i in range(n):
            if skip[i] or seqs[i] is None:
                continue

            gap = gaps[i]
            v_end[i] = airr_dict['v_germline_end'][i] + gap

            if self.chain == 'heavy' and airr_dict['d_sequence_start'][i] is not None:
                d_start[i] = airr_dict['d_sequence_start'][i] + gap
                d_end[i] = airr_dict['d_sequence_end'][i] + gap

            j_start[i] = airr_dict['j_sequence_start'][i] + gap
            j_end[i] = airr_dict['j_sequence_end'][i] + gap

        airr_dict['v_alignment_start'] = v_start
        airr_dict['v_alignment_end'] = v_end
        airr_dict['v_germline_start'] = v_start ## aligning the v germline start to 0 with the addition of gapps in the beginning of the sequence alignment
        airr_dict['v_germline_end'] = v_end ## add the gaps of the v germline to the end of the v segment
        airr_dict['d_alignment_start'] = d_start
        airr_dict['d_alignment_end'] = d_end
        airr_dict['j_alignment_start'] = j_start
        airr_dict['j_alignment_end'] = j_end

        return airr_dict
    
    
    def _populate_segment_alignment_columns(self, airr_dict):
        segments = ['v', 'j'] + (['d'] if self.chain == 'heavy' else [])
        n = len(airr_dict['sequence_alignment'])

        for seg in segments:
            s_col = f'{seg}_alignment_start'
            e_col = f'{seg}_alignment_end'

            airr_dict[f'{seg}_sequence_alignment'] = [
                airr_dict['sequence_alignment'][i][airr_dict[s_col][i]:airr_dict[e_col][i]]
                if not airr_dict['skip_processing'][i] and airr_dict[s_col][i] is not None and airr_dict[e_col][i] is not None else None
                for i in range(n)
            ]

            airr_dict[f'{seg}_sequence_alignment_aa'] = [
                airr_dict['sequence_alignment_aa'][i][airr_dict[s_col][i] // 3: airr_dict[e_col][i] // 3]
                if not airr_dict['skip_processing'][i] and airr_dict[s_col][i] is not None and airr_dict[e_col][i] is not None else None
                for i in range(n)
            ]

            airr_dict[f'{seg}_germline_alignment'] = [
                airr_dict['germline_alignment'][i][airr_dict[s_col][i]:airr_dict[e_col][i]]
                if not airr_dict['skip_processing'][i] and airr_dict[s_col][i] is not None and airr_dict[e_col][i] is not None else None
                for i in range(n)
            ]

            airr_dict[f'{seg}_germline_alignment_aa'] = [
                airr_dict['germline_alignment_aa'][i][airr_dict[s_col][i] // 3: airr_dict[e_col][i] // 3]
                if not airr_dict['skip_processing'][i] and airr_dict[s_col][i] is not None and airr_dict[e_col][i] is not None else None
                for i in range(n)
            ]
            
        return airr_dict

    
    def _add_region_columns(self, airr_dict):
        for region, (start, end) in self.regions.items():
            if region in ['cdr3', 'junction', 'fwr4']:
                continue

            airr_dict[region] = [
                seq[start:end] if seq is not None else None
                for seq in airr_dict['sequence_alignment']
            ]
            airr_dict[f'{region}_start'] = [start] * len(airr_dict['sequence_alignment'])
            airr_dict[f'{region}_end'] = [end] * len(airr_dict['sequence_alignment'])

            aa_start = None if start is None else start // 3
            aa_end = None if end is None else end // 3
            airr_dict[f'{region}_aa'] = [
                aa_seq[aa_start:aa_end] if aa_seq is not None else None
                for aa_seq in airr_dict['sequence_alignment_aa']
            ]

        return airr_dict

    
    def _add_cdr3_and_junction_columns(self, airr_dict):
        cols = {
            'junction': [], 'junction_aa': [], 'junction_length': [], 'junction_aa_length': [],
            'cdr3': [], 'cdr3_aa': [], 'cdr3_start': [], 'cdr3_end': [],
            'fwr4': [], 'fwr4_aa': [], 'fwr4_start': [], 'fwr4_end': []
        }

        for i in range(len(airr_dict['sequence_alignment'])):
            if airr_dict['skip_processing'][i] or airr_dict['sequence_alignment'][i] is None:
                for key in cols:
                    cols[key].append(None)
                continue

            j_call = airr_dict['j_call'][i].split(',')[0]
            j_anchor = self.j_anchor_dict.get(j_call, 0)
            offset = airr_dict['sequence_alignment'][i].count('.')

            junction_start = self.regions['junction'][0]
            junc_end = airr_dict['j_sequence_start'][i] - airr_dict['v_sequence_start'][i] + offset + j_anchor - airr_dict['j_germline_start'][i] + 3

            seq = airr_dict['sequence_alignment'][i]
            aa_seq = airr_dict['sequence_alignment_aa'][i]
            junction_nt = seq[junction_start:junc_end]
            junction_aa = aa_seq[junction_start // 3:junc_end // 3]
            cdr3_aa = junction_aa[1:-1] if len(junction_aa) > 2 else ''

            cdr3_start = junction_start + 3
            cdr3_end = junc_end - 3
            cdr3_nt = seq[cdr3_start:cdr3_end]

            fwr4_start = cdr3_end + 1
            fwr4_end = airr_dict['j_alignment_end'][i]
            fwr4_nt = seq[fwr4_start:fwr4_end]
            fwr4_aa = aa_seq[fwr4_start // 3:fwr4_end // 3]

            cols['junction'].append(junction_nt)
            cols['junction_aa'].append(junction_aa)
            cols['junction_length'].append(len(junction_nt))
            cols['junction_aa_length'].append(len(junction_aa))
            cols['cdr3'].append(cdr3_nt)
            cols['cdr3_aa'].append(cdr3_aa)
            cols['cdr3_start'].append(cdr3_start)
            cols['cdr3_end'].append(cdr3_end)
            cols['fwr4'].append(fwr4_nt)
            cols['fwr4_aa'].append(fwr4_aa)
            cols['fwr4_start'].append(fwr4_start)
            cols['fwr4_end'].append(fwr4_end)

        airr_dict.update(cols)
        return airr_dict
    
    
    def _add_np_regions(self, airr_dict):
        n = len(airr_dict['sequence'])

        if self.chain == 'heavy':
            airr_dict['np1'] = [
                airr_dict['sequence'][i][(airr_dict['v_sequence_end'][i]+1):airr_dict['d_sequence_start'][i]] # start one after v end
                if not airr_dict['skip_processing'][i] else None
                for i in range(n)
            ]
            airr_dict['np2'] = [
                airr_dict['sequence'][i][(airr_dict['d_sequence_end'][i]+1):airr_dict['j_sequence_start'][i]] # start one after d end
                if not airr_dict['skip_processing'][i] else None
                for i in range(n)
            ]
        else:
            airr_dict['np1'] = [
                airr_dict['sequence'][i][(airr_dict['v_sequence_end'][i]+1):airr_dict['j_sequence_start'][i]] # start one after v end
                if not airr_dict['skip_processing'][i] else None
                for i in range(n)
            ]
            airr_dict['np2'] = [None] * n

        airr_dict['np1_length'] = [len(x) if x is not None else None for x in airr_dict['np1']]
        airr_dict['np2_length'] = [len(x) if x is not None else None for x in airr_dict['np2']]

        return airr_dict
    
    
    def _add_productivity_flags(self, airr_dict):
        n = len(airr_dict['sequence'])

        airr_dict['stop_codon'] = [
            '*' in aa if aa is not None else False
            for aa in airr_dict['sequence_alignment_aa']
        ]

        airr_dict['vj_in_frame'] = [
            (airr_dict['cdr3_end'][i] - airr_dict['v_alignment_start'][i]) % 3 == 0 and
            (airr_dict['cdr3_end'][i] - airr_dict['cdr3_start'][i]) % 3 == 0 and
            not airr_dict['stop_codon'][i]
            if airr_dict['cdr3_end'][i] is not None and airr_dict['v_alignment_start'][i] is not None and airr_dict['cdr3_start'][i] is not None else False
            for i in range(n)
        ]

        return airr_dict
    
    
    def _reorder_and_finalize_columns(self, airr_dict):
        # Required and extra columns
        required_columns = [
            'sequence_id', 'sequence', 'locus', 'stop_codon', 'vj_in_frame', 'productive',
            'v_call', 'd_call', 'j_call', 'sequence_alignment', 'germline_alignment',
            'sequence_alignment_aa', 'germline_alignment_aa',
            'v_sequence_alignment', 'v_sequence_alignment_aa', 'v_germline_alignment',
            'v_germline_alignment_aa', 'd_sequence_alignment', 'd_sequence_alignment_aa',
            'd_germline_alignment', 'd_germline_alignment_aa', 'j_sequence_alignment',
            'j_sequence_alignment_aa', 'j_germline_alignment', 'j_germline_alignment_aa',
            'fwr1', 'fwr1_aa', 'cdr1', 'cdr1_aa', 'fwr2', 'fwr2_aa', 'cdr2', 'cdr2_aa',
            'fwr3', 'fwr3_aa', 'fwr4', 'fwr4_aa', 'cdr3', 'cdr3_aa', 'junction', 'junction_length',
            'junction_aa', 'junction_aa_length', 'v_sequence_start', 'v_sequence_end',
            'v_germline_start', 'v_germline_end', 'v_alignment_start', 'v_alignment_end', 
            'd_sequence_start', 'd_sequence_end', 'd_germline_start', 'd_germline_end', 
            'd_alignment_start', 'd_alignment_end','j_sequence_start', 'j_sequence_end',
            'j_germline_start', 'j_germline_end', 'j_alignment_start', 'j_alignment_end', 
            'fwr1_start', 'fwr1_end', 'cdr1_start', 'cdr1_end', 'fwr2_start', 'fwr2_end', 
            'cdr2_start', 'cdr2_end', 'fwr3_start', 'fwr3_end', 'fwr4_start', 'fwr4_end', 
            'cdr3_start', 'cdr3_end', 'np1', 'np1_length', 'np2', 'np2_length'
        ]
        extra_columns = ['v_likelihoods', 'd_likelihoods', 'j_likelihoods', 'mutation_rate', 'ar_indels']

        # Fill missing columns with None
        for col in required_columns + extra_columns:
            if col not in airr_dict:
                airr_dict[col] = [None] * len(airr_dict['sequence'])

        # 1-based indexing
        for col in airr_dict:
            if col.endswith('_start') or col.endswith('_end'):
                airr_dict[col] = [x + 1 for x in airr_dict[col]]

        # Convert booleans to 'T'/'F'
        for col in ['stop_codon', 'vj_in_frame', 'productive']:
            if col in airr_dict:
                airr_dict[col] = [
                    'T' if v else 'F' if v is not None else None
                    for v in airr_dict[col]
                ]
                
        # Remove skip_processing
        airr_dict.pop('skip_processing', None)

        return {col: airr_dict[col] for col in required_columns + extra_columns}
    
    def build_dataframe(self):
        po = self.predict_object
        n = len(po.sequences)

        # Extract sequence IDs from the input file
        file_path = po.file_info.path
        file_type = po.file_info.file_type
        sequence_ids = self._extract_sequence_ids(file_path, file_type)
        
        metadata_data = {}
        if sequence_ids:
            metadata_data = self._parse_sequence_id(sequence_ids)
            po.metadata = pd.DataFrame(metadata_data)

            
        airr_dict = {
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
            'mutation_rate': po.processed_predictions.get('mutation_rate', [None]*n),
            'ar_indels': po.processed_predictions.get('indel_count', [None]*n),
            'v_likelihoods': po.likelihoods_of_selected_alleles.get('v', [None]*n),
            'j_likelihoods': po.likelihoods_of_selected_alleles.get('j', [None]*n),
            'd_likelihoods': po.likelihoods_of_selected_alleles.get('d', [None]*n) if self.chain == 'heavy' else [None]*n,
        }

        if self.chain == 'heavy':
            airr_dict['d_sequence_start'] = [i['start_in_seq'] for i in po.germline_alignments['d']]
            airr_dict['d_sequence_end'] = [i['end_in_seq'] for i in po.germline_alignments['d']]
            airr_dict['d_germline_start'] = [abs(i['start_in_ref']) for i in po.germline_alignments['d']]
            airr_dict['d_germline_end'] = [i['end_in_ref'] for i in po.germline_alignments['d']]
            airr_dict['d_call'] = [','.join(i) for i in po.selected_allele_calls['d']]
            airr_dict['locus'] = ['IGH'] * n
        else:
            airr_dict['d_sequence_start'] = [None] * n
            airr_dict['d_sequence_end'] = [None] * n
            airr_dict['d_germline_start'] = [None] * n
            airr_dict['d_germline_end'] = [None] * n
            airr_dict['d_call'] = [''] * n
            airr_dict['locus'] = ['IGK' if i == 1 else 'IGL' for i in po['type_'].astype(int).squeeze()]
              
        airr_dict = self._get_alignments(airr_dict, n)
        airr_dict = self._translate_alignments(airr_dict)
        airr_dict = self._map_segment_alignment_positions(airr_dict)
        airr_dict = self._populate_segment_alignment_columns(airr_dict)
        airr_dict = self._add_region_columns(airr_dict)
        airr_dict = self._add_cdr3_and_junction_columns(airr_dict)
        airr_dict = self._add_np_regions(airr_dict)
        airr_dict = self._add_productivity_flags(airr_dict)
        airr_dict = self._reorder_and_finalize_columns(airr_dict)
        
        for key, value in metadata_data.items():
            airr_dict[key] = value
            
        cols = ['sequence_id'] + [col for col in airr_dict if col != 'sequence_id']
        airr_dict = {col: airr_dict[col] for col in cols}
        
        return pd.DataFrame(airr_dict)