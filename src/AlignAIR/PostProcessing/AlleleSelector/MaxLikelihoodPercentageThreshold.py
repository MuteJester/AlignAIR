from typing import Union

import numpy as np
from GenAIRR.dataconfig import DataConfig
from joblib import delayed, Parallel
from tqdm.auto import tqdm

from AlignAIR.Data import MultiDataConfigContainer
from AlignAIR.Data.encoders import AlleleEncoder


def deterministic_coef_method(likelihoods,th=0.1):
    max_index = np.argmax(likelihoods)
    max_likelihood = likelihoods[max_index]
    threshold_value = max_likelihood * th
    indices = np.where(likelihoods >= threshold_value)[0]
    indices = indices[np.argsort(-likelihoods[indices])]
    indices = indices[:3] if len(indices) > 3 else indices
    return np.array(indices.tolist()), len(indices)




class MaxLikelihoodPercentageThreshold:
    def __init__(self, dataconfig:Union[DataConfig, MultiDataConfigContainer]):

        self.dataconfig = dataconfig
        self.allele_encoder = AlleleEncoder()

        self.add_allele_dictionaries()
        self.register_alleles_to_ohe()

        self.properties_map = self.allele_encoder.get_properties_map()

    @property
    def has_d(self):
        if isinstance(self.dataconfig, DataConfig):
            return self.dataconfig.metadata.has_d
        elif isinstance(self.dataconfig, MultiDataConfigContainer):
            return self.dataconfig.has_at_least_one_d()
        else:
            raise ValueError("dataconfig must be either a DataConfig or MultiDataConfigContainer instance")

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
            results.append(([self.properties_map[allele.upper()]['reverse_mapping'][i] for i in selected_alleles_index],likelihoods))

        return results

