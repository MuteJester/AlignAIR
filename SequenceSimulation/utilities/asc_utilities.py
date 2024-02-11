import pandas as pd
import numpy as np
from collections import defaultdict
from SequenceSimulation.utilities.data_utilities import create_allele_dict
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from SequenceSimulation.alleles.allele import VAllele

def allele_diff(reference_allele, sample_allele, position_threshold=0, snps=True):
    germs = [reference_allele, sample_allele]
    max_length = max(len(germ) for germ in germs)
    for i in range(len(germs)):
        germs[i] += '.' * (max_length - len(germs[i]))
    def setdiff_mat(x):
        unique_chars = set(x)
        filter_chars = {'.', 'N', '-'}
        return len(unique_chars - filter_chars)
    idx_strings = []
    for i in range(max_length):
        column_chars = [germ[i] for germ in germs]
        diff_count = setdiff_mat(column_chars)
        if diff_count > 1 and i >= (position_threshold-1):
            if snps:
                concatenated_str = column_chars[0].upper() + str(i+1) + column_chars[1].upper()
                idx_strings.append(concatenated_str)
            else:
                idx_strings.append(i+1)
    return idx_strings

def hamming_distance(s1, s2):
    s1 = np.array(list(s1))
    s2 = np.array(list(s2))
    mismatches = np.sum((s1 != 'N') & (s2 != 'N') & (s1 != s2))
    gapGapMatches = np.sum((s1 == '.') & (s2 == '.'))
    count = len(s1) - gapGapMatches    
    distance = mismatches / count if count > 0 else 0
    return distance

""" def asc_distance(germline_set):
    # Check sequences length. If not even pad the sequences to the max length
    max_length = max(len(s) for s in germline_set.values())
    germline_set_padded = {allele: sequence.ljust(max_length, 'N') for allele, sequence in germline_set.items()}
    # Compute the distance between pairs. penalize for gaps
    germline_distance = np.zeros((len(germline_set_padded), len(germline_set_padded)))
    sequences = list(germline_set_padded.values())
    for i in range(len(sequences)):
        s1 = np.array(list(sequences[i]))
        germline_distance[i, :] = np.array([hamming_distance(s1, np.array(list(s2))) for s2 in sequences])
    return germline_distance """

def asc_distance(germline_set, trim_3prime_side=318):
    if trim_3prime_side is not None:
        germline_set = {allele: seq[:trim_3prime_side] for allele, seq in germline_set.items()}
    max_length = max(len(s) for s in germline_set.values())
    germline_set_padded = {allele: sequence.ljust(max_length, 'N') for allele, sequence in germline_set.items()}
    
    sequences_array = np.array([list(seq) for seq in germline_set_padded.values()])

    num_sequences = len(sequences_array)
    germline_distance_matrix = np.zeros((num_sequences, num_sequences))
    
    for i in range(num_sequences):
        s1 = sequences_array[i]
        for j in range(num_sequences):
            s2 = sequences_array[j]
            mismatches = np.sum((s1 != 'N') & (s2 != 'N') & (s1 != s2))
            gapGapMatches = np.sum((s1 == '.') & (s2 == '.'))
            count = len(s1) - gapGapMatches
            germline_distance_matrix[i, j] = mismatches / count if count > 0 else 0

    return germline_distance_matrix

def asc_clust(germline_distance, germline_set, family_threshold = 75, allele_cluster_threshold = 95, trim_3prime_side = 318):    
    condensed_distance = squareform(germline_distance)
    germline_cluster = linkage(condensed_distance, method='complete')
    dendro = dendrogram(germline_cluster, labels=list(germline_set.keys()), no_plot=True)
    labels_order = dendro['ivl']
    leaf_order = dendro['leaves']
    
    labels = list(germline_set.keys())
    segment = labels[0][0:4]
    
    thresh = [1-family_threshold/100,1-allele_cluster_threshold/100]
    family_cluster = fcluster(germline_cluster, 
                        t=thresh[0], 
                        criterion='distance')
    family_clusters_in_order = family_cluster[leaf_order]
    family_clusters_in_order_renumbered = pd.factorize(family_clusters_in_order)[0] + 1
    
    allele_cluster = fcluster(germline_cluster, 
                            t=thresh[1], 
                            criterion='distance')
    allele_clusters_in_order = allele_cluster[leaf_order]
    allele_clusters_in_order_renumbered = pd.factorize(allele_clusters_in_order)[0] + 1
    
    alleleClusterTable = pd.DataFrame({
                'Family': family_clusters_in_order_renumbered,
                'Allele_Cluster': allele_clusters_in_order_renumbered,
                'Allele': labels_order
            })
    
    alleleClusterTable['duplicated_allele'] = pd.NA
    alleleClusterTable['diff_pos_past_trim'] = pd.NA
    
    for i, row in enumerate(germline_cluster):
        cluster_1, cluster_2, distance, _ = row
        if distance == 0:
            # check which allele is longer, this should be the reference allele
            if len(germline_set[labels[int(cluster_1)]]) > len(germline_set[labels[int(cluster_2)]]):
                cluster_1_label = labels[int(cluster_1)]
                cluster_2_label = labels[int(cluster_2)]
            else:
                cluster_2_label = labels[int(cluster_1)]
                cluster_1_label = labels[int(cluster_2)]
            # check if the 'duplicated allele' is seperated passed the 3prime trim
            if trim_3prime_side is not None:
                snps = allele_diff(germline_set[cluster_1_label], germline_set[cluster_2_label], trim_3prime_side)
                if snps:
                    alleleClusterTable.loc[alleleClusterTable['Allele'] == cluster_2_label, 'diff_pos_past_trim'] = ('_').join(map(str, snps))
                    alleleClusterTable.loc[alleleClusterTable['Allele'] == cluster_1_label, 'duplicated_allele'] = cluster_1_label+","+cluster_2_label
                    alleleClusterTable.loc[alleleClusterTable['Allele'] == cluster_2_label, 'duplicated_allele'] = cluster_1_label+","+cluster_2_label
                else:
                    duplicate_cell_is_na = pd.isna(alleleClusterTable['duplicated_allele'].loc[alleleClusterTable['Allele'] == cluster_1_label]).any()
                    # check if the duplicated allele is already set
                    if ~duplicate_cell_is_na:
                        alleleClusterTable.loc[alleleClusterTable['Allele'] == cluster_1_label, 'duplicated_allele'] += cluster_2_label
                    else:
                        alleleClusterTable.loc[alleleClusterTable['Allele'] == cluster_1_label, 'duplicated_allele'] = cluster_2_label
                        alleleClusterTable.loc[alleleClusterTable['Allele'] == cluster_2_label, 'duplicated_allele'] = 'remove'
            else:
                duplicate_cell_is_na = pd.isna(alleleClusterTable['duplicated_allele'].loc[alleleClusterTable['Allele'] == cluster_1_label]).any()
                # check if the duplicated allele is already set
                if ~duplicate_cell_is_na:
                    alleleClusterTable.loc[alleleClusterTable['Allele'] == cluster_1_label, 'duplicated_allele'] += cluster_2_label
                else:
                    alleleClusterTable.loc[alleleClusterTable['Allele'] == cluster_1_label, 'duplicated_allele'] = cluster_2_label
                    alleleClusterTable.loc[alleleClusterTable['Allele'] == cluster_2_label, 'duplicated_allele'] = 'remove'
    
    # rename the alleles (segment-FamilyCluster-AlleleCluster*allele)
    alleleClusterTable_copy = alleleClusterTable.copy()
    alleleClusterTable_copy = alleleClusterTable_copy[alleleClusterTable_copy['duplicated_allele'] != 'remove']
    alleleClusterTable_copy['allele_idx'] = alleleClusterTable_copy.groupby('Allele_Cluster')['Family'].transform(lambda x: range(1, len(x)+1))
    alleleClusterTable_copy['allele_index'] = alleleClusterTable_copy['allele_idx'].copy()
    # Change 'allele_index' value to the minimum for rows with the same 'duplicated_allele'
    alleleClusterTable_copy['allele_index'] = (
        alleleClusterTable_copy.groupby(['Allele_Cluster', 'Family', 'duplicated_allele'])
        ['allele_idx'].transform('min')
    ).fillna(alleleClusterTable_copy['allele_idx'])
    alleleClusterTable_copy['allele_index'] = alleleClusterTable_copy.groupby('Allele_Cluster')['allele_index'].apply(lambda x: x.diff().fillna(1).cumsum())
    alleleClusterTable_copy = alleleClusterTable_copy.drop('allele_idx', axis=1)
    alleleClusterTable_copy['new_allele'] = alleleClusterTable_copy.apply(lambda row: f"{segment}F{row['Family']}-G{row['Allele_Cluster']}*{str(int(row['allele_index'])).zfill(2) if row['allele_index'] >= 1 else str(row['allele_index'])}", axis=1)
    alleleClusterTable_copy['new_allele'] += alleleClusterTable_copy['diff_pos_past_trim'].apply(lambda x: f"_{x}" if pd.notna(x) else "")
    return alleleClusterTable_copy

def asc_dict(allele_cluster_table, germline_set):
                        
        germline_set_asc = defaultdict(list)
        
        for i, row in allele_cluster_table.iterrows():
            allele = row['Allele']
            new_allele = row['new_allele']
            # get asc
            asc = new_allele.split("*")[0]
            seq = germline_set[allele]
            ungapped_length = len(seq.replace(".",""))
            germline_set_asc[asc].append(VAllele(new_allele, seq, ungapped_length))
                
        return germline_set_asc
    
def create_asc_germline_set(user_reference, segment = "V", trim_3prime_side = 318, family_threshold = 75, allele_cluster_threshold = 95):
    
    germline_set_user = defaultdict(dict)
    for key, value in create_allele_dict(user_reference).items():
        for val in value:
            if segment in val.name:
                germline_set_user[val.name] = val.gapped_seq
    
    germline_distance = asc_distance(germline_set_user)

    asc_table = asc_clust(germline_distance, germline_set_user, family_threshold, allele_cluster_threshold, trim_3prime_side)

    germline_set_asc = asc_dict(asc_table, germline_set_user)
    
    return germline_set_asc,asc_table
    
def test_asc(fasta):
    germline_set = create_allele_dict(fasta)
    germline_set_asc = create_asc_germline_set(germline_set)
    return germline_set_asc

def profile(fasta):
    from cProfile import Profile
    germline_set = create_allele_dict(fasta)
    cProfile.run('create_asc_germline_set(germline_set)')
    with Profile() as profile:
        print(f"{fib(35) = }")
        (
            Stats(profile)
            .strip_dirs()
            .sort_stats(SortKey.CALLS)
            .print_stats()
        )