import numpy as np
from GenAIRR.dataconfig import DataConfig
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from AlignAIR.Data.encoders import AlleleEncoder


class CappedDynamicConfidenceThreshold:
    def __init__(self, dataconfig:DataConfig):

        self.dataconfig = dataconfig
        self.allele_encoder = AlleleEncoder()

        self.chain = dataconfig.metadata.chain_type
        self.add_allele_dictionaries()
        self.register_alleles_to_ohe()

        self.properties_map = self.allele_encoder.get_properties_map()


    def add_allele_dictionaries(self):

        self.v_dict = {i.name: i for i in self.dataconfig.allele_list('v')}
        self.j_dict = {i.name: i for i in self.dataconfig.allele_list('j')}

        if self.has_d:
            self.d_dict = {i.name: i for i in self.dataconfig.allele_list('d')}

    def register_alleles_to_ohe(self):
        """
        Register alleles to the one-hot encoder based on the derived dictionaries.
        Returns:

        """
        v_alleles = sorted(list(self.v_dict))
        j_alleles = sorted(list(self.j_dict))

        self.v_allele_count = len(v_alleles)
        self.j_allele_count = len(j_alleles)

        self.allele_encoder.register_gene("V", v_alleles, sort=False)
        self.allele_encoder.register_gene("J", j_alleles, sort=False)

        if self.has_d:
            d_alleles = sorted(list(self.d_dict)) + ['Short-D']
            self.d_allele_count = len(d_alleles)
            self.allele_encoder.register_gene("D", d_alleles, sort=False)

    def __getitem__(self, gene):
        return self.properties_map[gene.upper()]['allele_call_ohe']

    def dynamic_cumulative_confidence_threshold(self, prediction, percentage=0.9,cap=3):
        sorted_indices = np.argsort(prediction)[::-1]
        selected_labels = []
        likelihoods = []
        cumulative_confidence = 0.0

        total_confidence = sum(prediction)
        threshold = percentage * total_confidence

        for idx in sorted_indices:
            cumulative_confidence += prediction[idx]
            selected_labels.append(idx)
            likelihoods.append(prediction[idx])

            if cumulative_confidence >= threshold or len(selected_labels) >= cap:
                break

        return selected_labels,likelihoods

    def get_alleles_mt(self, likelihood_vectors, confidence=0.9, allele='v', n_process=1,cap=3):
        def process_vector(vec):
            selected_alleles_index,likelihoods = self.dynamic_cumulative_confidence_threshold(vec, percentage=confidence,cap=cap)
            return [self[allele][i] for i in selected_alleles_index],likelihoods

        results = Parallel(n_jobs=n_process)(delayed(process_vector)(vec) for vec in tqdm(likelihood_vectors))
        return results

    def get_alleles(self, likelihood_vectors, confidence=0.9, allele='v',cap=3,verbose=False):
        results = []
        desc = f'Processing {allele.upper()} Likelihoods'
        if verbose:
            iterator = tqdm(likelihood_vectors,desc=desc)
        else:
            iterator = likelihood_vectors

        for vec in iterator:
            selected_alleles_index,likelihoods = self.dynamic_cumulative_confidence_threshold(vec, percentage=confidence,cap=cap)
            results.append(([self[allele][i] for i in selected_alleles_index],likelihoods))

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
