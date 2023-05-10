from VDeepJUnbondedDataset import global_genotype
import pandas as pd
import re
import json


def create_sub_classes_dict(v_dict, d_dict, j_dict):
    def nested_get_sub_callses(d, v_d_or_j):
        sub_cllases = {}
        # V or D
        if v_d_or_j in ["v", "d"]:
            for call in d.values():
                fam, gene, allele = call["family"], call["gene"], call["allele"]
                if fam not in sub_cllases.keys():
                    sub_cllases[fam] = {}
                    sub_cllases[fam][gene] = {allele: allele}
                else:
                    if gene not in sub_cllases[fam].keys():
                        sub_cllases[fam][gene] = {allele: allele}
                    else:
                        sub_cllases[fam][gene][allele] = allele
        # J
        elif v_d_or_j == "j":
            for call in d.values():
                gene, allele = call["gene"], call["allele"]
                if gene not in sub_cllases.keys():
                    sub_cllases[gene] = {allele: allele}
                else:
                    sub_cllases[gene][allele] = allele
        return sub_cllases

    sub_cllases_v = nested_get_sub_callses(v_dict, "v")
    sub_cllases_d = nested_get_sub_callses(d_dict, "d")
    sub_cllases_j = nested_get_sub_callses(j_dict, "j")

    sub_cllases = {"V": sub_cllases_v, "D": sub_cllases_d, "J": sub_cllases_j}

    return sub_cllases


def derive_call_dictionaries(locus):
    v_dict, d_dict, j_dict = dict(), dict(), dict()
    for call in ["V", "D", "J"]:
        for idx in range(2):
            for N in locus[idx][call]:
                if call == "V":
                    family, G = N.name.split("-", 1)
                    gene, allele = G.split("*")
                    v_dict[N.name] = {
                        "family": family,
                        "gene": gene,
                        "allele": allele,
                    }
                elif call == "D":
                    family, G = N.name.split("-", 1)
                    gene, allele = G.split("*")
                    d_dict[N.name] = {
                        "family": family,
                        "gene": gene,
                        "allele": allele,
                    }
                elif call == "J":
                    gene, allele = N.name.split("*")
                    j_dict[N.name] = {"gene": gene, "allele": allele}

    return v_dict, d_dict, j_dict


locus = global_genotype()


# v_call = [call.name for call in locus[0]["V"]]
# d_call = [call.name for call in locus[0]["D"]]
# j_call = [call.name for call in locus[0]["J"]]
# v_call = pd.Series(v_call)
# d_call = pd.Series(d_call)
# j_call = pd.Series(j_call)

# v_dict, d_dict, j_dict = derive_call_dictionaries(v_call, d_call, j_call, locus)
v_dict, d_dict, j_dict = derive_call_dictionaries(locus)

sub_cllases = create_sub_classes_dict(v_dict, d_dict, j_dict)

# Save the dictionary to a file
with open("airrship/data/sub_cllases.json", "w") as f:
    json.dump(sub_cllases, f)
