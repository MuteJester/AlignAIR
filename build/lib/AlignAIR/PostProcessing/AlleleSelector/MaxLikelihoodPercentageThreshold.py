import numpy as np
from joblib import delayed, Parallel
from tqdm.auto import tqdm


def deterministic_coef_method(likelihoods,th=0.1):
    max_index = np.argmax(likelihoods)
    max_likelihood = likelihoods[max_index]
    threshold_value = max_likelihood * th
    indices = np.where(likelihoods >= threshold_value)[0]
    indices = indices[np.argsort(-likelihoods[indices])]
    indices = indices[:3] if len(indices) > 3 else indices
    return np.array(indices.tolist()), len(indices)




class MaxLikelihoodPercentageThreshold:
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

            V = self.v_kappa_dict.copy()
            V.update(self.v_lambda_dict)
            J = self.j_kappa_dict.copy()
            J.update(self.j_lambda_dict)
            self.reference_map = {
                'v': V,
                'j': J,
            }
        else:
            self.v_heavy_dict = {j.name: j.ungapped_seq.upper() for i in self.heavy_dataconfig.v_alleles for j in
                                 self.heavy_dataconfig.v_alleles[i]}
            self.d_heavy_dict = {j.name: j.ungapped_seq.upper() for i in self.heavy_dataconfig.d_alleles for j in
                                 self.heavy_dataconfig.d_alleles[i]}
            self.j_heavy_dict = {j.name: j.ungapped_seq.upper() for i in self.heavy_dataconfig.j_alleles for j in
                                 self.heavy_dataconfig.j_alleles[i]}

            self.reference_map = {
                'v': self.v_heavy_dict,
                'd': self.d_heavy_dict,
                'j': self.j_heavy_dict,
            }

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

    def max_likelihood_percentage_threshold(self, prediction, percentage=0.21,cap=3):
        max_index = np.argmax(prediction)
        max_likelihood = prediction[max_index]
        threshold_value = max_likelihood * percentage
        indices = np.where(prediction >= threshold_value)[0]
        indices = indices[np.argsort(-prediction[indices])]
        indices = indices[:cap] if len(indices) > cap else indices
        likelihoods = prediction[indices]
        return indices, likelihoods

    def get_alleles_mt(self, likelihood_vectors, percentage=0.21, allele='v', n_process=1,cap=3):
        def process_vector(vec):
            selected_alleles_index,likelihoods = self.get_alleles_mt(vec, percentage=percentage,cap=cap)
            return [self[allele][i] for i in selected_alleles_index],likelihoods

        results = Parallel(n_jobs=n_process)(delayed(process_vector)(vec) for vec in tqdm(likelihood_vectors))
        return results

    def get_alleles(self, likelihood_vectors, percentage=0.21, allele='v',cap=3,verbose=False):
        results = []
        desc = f'Processing {allele.upper()} Likelihoods'
        if verbose:
            iterator = tqdm(likelihood_vectors,desc=desc)
        else:
            iterator = likelihood_vectors

        for vec in iterator:
            selected_alleles_index,likelihoods = self.max_likelihood_percentage_threshold(vec, percentage=percentage,cap=cap)
            results.append(([self[allele][i] for i in selected_alleles_index],likelihoods))

        return results

