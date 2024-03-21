from joblib import Parallel, delayed
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import jaccard_score


class DynamicConfidenceThreshold:
    def __init__(self, heavy_dataconfig=None, kappa_dataconfig=None, lambda_dataconfig=None):

        self.heavy_dataconfig = heavy_dataconfig
        self.kappa_dataconfig = kappa_dataconfig
        self.lambda_dataconfig = lambda_dataconfig

        self.chain = self.determine_chain()
        self.derive_allele_dictionaries()
        self.derive_call_one_hot_representation()

    def determine_chain(self):
        if self.heavy_dataconfig is not None:
            return 'heavy'
        elif self.kappa_dataconfig is not None and self.lambda_dataconfig is not None:
            return 'light'
        else:
            raise ValueError("Invalid chain configuration")

    def derive_allele_dictionaries(self):

        if self.chain == 'light':
            self.v_kappa_dict = {j.name: j.ungapped_seq.upper() for i in self.kappa_dataconfig.v_alleles for j in
                                 self.kappa_dataconfig.v_alleles[i]}
            self.j_kappa_dict = {j.name: j.ungapped_seq.upper() for i in self.kappa_dataconfig.j_alleles for j in
                                 self.kappa_dataconfig.j_alleles[i]}

            self.v_lambda_dict = {j.name: j.ungapped_seq.upper() for i in self.lambda_dataconfig.v_alleles for j in
                                  self.lambda_dataconfig.v_alleles[i]}
            self.j_lambda_dict = {j.name: j.ungapped_seq.upper() for i in self.lambda_dataconfig.j_alleles for j in
                                  self.lambda_dataconfig.j_alleles[i]}
        else:
            self.v_heavy_dict = {j.name: j.ungapped_seq.upper() for i in self.heavy_dataconfig.v_alleles for j in
                                 self.heavy_dataconfig.v_alleles[i]}
            self.d_heavy_dict = {j.name: j.ungapped_seq.upper() for i in self.heavy_dataconfig.d_alleles for j in
                                 self.heavy_dataconfig.d_alleles[i]}
            self.j_heavy_dict = {j.name: j.ungapped_seq.upper() for i in self.heavy_dataconfig.j_alleles for j in
                                 self.heavy_dataconfig.j_alleles[i]}

    def derive_call_one_hot_representation(self):
        if self.chain == 'light':

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
        else:
            v_alleles = sorted(list(self.v_heavy_dict))
            d_alleles = sorted(list(self.d_heavy_dict))
            d_alleles = d_alleles + ['Short-D']
            j_alleles = sorted(list(self.j_heavy_dict))

            v_allele_count = len(v_alleles)
            d_allele_count = len(d_alleles)
            j_allele_count = len(j_alleles)

            v_allele_call_ohe = {i: f for i, f in enumerate(v_alleles)}
            d_allele_call_ohe = {i: f for i, f in enumerate(d_alleles)}
            j_allele_call_ohe = {i: f for i, f in enumerate(j_alleles)}

            self.properties_map = {
                "V": {"allele_count": v_allele_count, "allele_call_ohe": v_allele_call_ohe},
                "D": {"allele_count": d_allele_count, "allele_call_ohe": d_allele_call_ohe},
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

    def agreement_score(self, allele_true, allele_pred):
        """
        Calculate the agreement score between true and predicted alleles.
        """
        set1 = set(allele_true)
        set2 = set(allele_pred)
        intersection = set1.intersection(set2)
        union = set1.union(set2)

        if not union:
            return 0.0  # Avoid division by zero if both lists are empty

        jaccard_index = len(intersection) / len(union)
        return jaccard_index

    def agreement(self, allele_true, allele_pred):
        return len(set(allele_true) & set(allele_pred)) > 0

    def optimize_confidence(self, likelihood_vectors, ground_truth, allele='v', n_process=1, steps=10):
        """
        Find the optimal confidence value.
        """
        best_confidence = 0
        best_score = 0
        best_allele_count = float('inf')

        for confidence in np.linspace(0, 1, steps):
            predicted_alleles = self.get_alleles(likelihood_vectors, confidence, allele, n_process)
            agreement_scores = []

            for pred, true in zip(predicted_alleles, ground_truth):
                score = self.agreement_score(true, pred)
                agreement_scores.append(score)

            average_score = np.mean(agreement_scores)
            average_allele_count = np.mean([len(a) for a in predicted_alleles])

            if average_score > best_score or (average_score == best_score and average_allele_count < best_allele_count):
                best_score = average_score
                best_confidence = confidence
                best_allele_count = average_allele_count

        return best_confidence, best_score, best_allele_count

    def get_confidence_range(self, likelihood_vectors, ground_truth, allele='v', n_process=1, steps=10):
        agg_results = dict()
        for confidence in np.linspace(0, 1, steps):
            predicted_alleles = self.get_alleles(likelihood_vectors, confidence, allele, n_process)
            agreement_scores = []
            calls = []
            agreements = []

            for pred, true in zip(predicted_alleles, ground_truth):
                score = self.agreement_score(true, pred)

                agreements.append(self.agreement(true, pred))
                agreement_scores.append(score)
                calls.append(len(pred))
            agg_results[confidence] = {'agreement_scores': agreement_scores,
                                       'agreements': agreements,
                                       'calls': calls}

        return agg_results
