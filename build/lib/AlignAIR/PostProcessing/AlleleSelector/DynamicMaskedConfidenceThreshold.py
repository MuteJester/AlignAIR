from joblib import Parallel, delayed
import numpy as np
from tqdm.auto import tqdm

class DynamicMaskedConfidenceThreshold:
    def __init__(self, kappa_dataconfig,lambda_dataconfig):
        self.kappa_dataconfig = kappa_dataconfig
        self.lambda_dataconfig = lambda_dataconfig

        self.derive_allele_dictionaries()

        self.derive_call_one_hot_representation()



    def derive_allele_dictionaries(self):
        self.v_kappa_dict = {j.name: j.ungapped_seq.upper() for i in self.kappa_dataconfig.v_alleles for j in
                             self.kappa_dataconfig.v_alleles[i]}
        self.j_kappa_dict = {j.name: j.ungapped_seq.upper() for i in self.kappa_dataconfig.j_alleles for j in
                             self.kappa_dataconfig.j_alleles[i]}

        self.v_lambda_dict = {j.name: j.ungapped_seq.upper() for i in self.lambda_dataconfig.v_alleles for j in
                              self.lambda_dataconfig.v_alleles[i]}
        self.j_lambda_dict = {j.name: j.ungapped_seq.upper() for i in self.lambda_dataconfig.j_alleles for j in
                              self.lambda_dataconfig.j_alleles[i]}

    def derive_call_one_hot_representation(self):
        v_alleles = sorted(list(self.v_kappa_dict)) + sorted(list(self.v_lambda_dict))
        j_alleles = sorted(list(self.j_kappa_dict)) + sorted(list(self.j_lambda_dict))

        v_allele_count = len(v_alleles)
        j_allele_count = len(j_alleles)

        v_allele_call_ohe = {i: f for i, f in enumerate(v_alleles)}
        j_allele_call_ohe = {i: f for i, f in enumerate(j_alleles)}

        self.properties_map = {
            "V": {"allele_count": v_allele_count, "allele_call_ohe": v_allele_call_ohe},
            "J": {"allele_count": j_allele_count, "allele_call_ohe": j_allele_call_ohe},
        }

    def __getitem__(self, gene):
        return self.properties_map[gene.upper()]['allele_call_ohe']

    def dynamic_cumulative_confidence_threshold(self, prediction, percentage=0.9):
        sorted_indices = np.argsort(prediction)[::-1]
        selected_labels = []
        cumulative_confidence = 0.0

        total_confidence = sum(prediction)
        threshold = percentage * total_confidence

        for idx in sorted_indices:
            cumulative_confidence += prediction[idx]
            selected_labels.append(idx)

            if cumulative_confidence >= threshold:
                break

        return selected_labels

    def get_alleles(self, likelihood_vectors, confidence=0.9, allele='v', n_process=1):
        def process_vector(vec):
            selected_alleles_index = self.dynamic_cumulative_confidence_threshold(vec, percentage=confidence)
            return [self[allele][i] for i in selected_alleles_index]

        results = Parallel(n_jobs=n_process)(delayed(process_vector)(vec) for vec in tqdm(likelihood_vectors))
        return results

