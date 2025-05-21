import numpy as np
from GenAIRR.dataconfig import DataConfig

from AlignAIR.PredictObject.PredictObject import PredictObject
from AlignAIR.Step.Step import Step
from AlignAIR.Utilities.GenotypeYamlParser import GenotypeYamlParser
from AlignAIR.Utilities.step_utilities import DataConfigLibrary
from tqdm.auto import tqdm

class IsInGenotype:
    def __init__(self, data_config):
        if type(data_config) is DataConfig:
            if 'v_alleles' in hasattr(data_config, 'v_alleles'):
                self.v_alleles = {i.name for j in data_config.v_alleles for i in data_config.v_alleles[j]}
            if 'd_alleles' in hasattr(data_config, 'd_alleles'):
                self.d_alleles = {i.name for j in data_config.d_alleles for i in data_config.d_alleles[j]}
            if 'j_alleles' in hasattr(data_config, 'j_alleles'):
                self.j_alleles = {i.name for j in data_config.j_alleles for i in data_config.j_alleles[j]}
        elif type(data_config) is DataConfigLibrary:
            self.v_alleles = data_config.reference_allele_names('v')
            self.d_alleles = data_config.reference_allele_names('d')
            self.j_alleles = data_config.reference_allele_names('j')
        elif type(data_config) is GenotypeYamlParser:
            self.v_alleles = data_config.v
            self.d_alleles = data_config.d
            self.j_alleles = data_config.j
        else:
            raise ValueError("data_config must be of type DataConfig or DataConfigLibrary")

    def __getitem__(self, key):
        allele_type = 'V' if 'V' in key else 'D' if 'D' in key else 'J'
        return key in getattr(self, f"{allele_type.lower()}_alleles")

    def is_in_genotype(self, allele):
        return self[allele]


class GenotypeBasedLikelihoodAdjustmentStep(Step):

    def __init__(self, name):
        super().__init__(name)
        self.v_allele_name_to_index = None
        self.d_allele_name_to_index = None
        self.j_allele_name_to_index = None

        self.v_allele_index_to_name = None
        self.d_allele_index_to_name = None
        self.j_allele_index_to_name = None
        self.genotype_checker = None

    def is_in_genotype(self, allele):
        return self.genotype_checker[allele]

    def genotype_alleles(self, predicted_alleles):
        genotype_alleles = {allele: likelihood for allele, likelihood in predicted_alleles.items() if
                            self.is_in_genotype(allele)}
        return genotype_alleles

    def non_genotype_alleles(self, predicted_alleles):
        non_genotype_alleles = {allele: likelihood for allele, likelihood in predicted_alleles.items() if
                                not self.is_in_genotype(allele)}
        return non_genotype_alleles

    def bounded_redistribution(self, predicted_alleles):
        """
        Removes non-genotype alleles and reweights the likelihoods, ensuring all reweighted likelihoods
        lie between 0 and 1 but do not need to sum to 1.

        :param predicted_alleles: dict with allele names as keys and likelihoods as values.
        :param genotype_checker: object with __getitem__ that returns True if an allele is in the genotype, False otherwise.
        :return: dict with reweighted alleles and their likelihoods.
        """
        # Separate genotype and non-genotype alleles
        genotype_alleles = self.genotype_alleles(predicted_alleles)
        non_genotype_alleles = self.non_genotype_alleles(predicted_alleles)

        # Calculate total likelihoods
        total_genotype_likelihood = sum(genotype_alleles.values())
        total_non_genotype_likelihood = sum(non_genotype_alleles.values())

        # Redistribute non-genotype likelihoods proportionally to genotype alleles
        if total_genotype_likelihood > 0:
            redistribution_factor = total_non_genotype_likelihood / total_genotype_likelihood
            genotype_alleles = {
                allele: min(1, likelihood + likelihood * redistribution_factor)
                for allele, likelihood in genotype_alleles.items()
            }

        return genotype_alleles

    def initialize_allele_index_translation_maps(self, dataconfig_library: DataConfigLibrary):
        """
        Initializes the allele name to index and index to name mappings for each gene.
        Args:
            dataconfig_library:

        Returns:
        """

        # sorted lists of allele name for each gene just like the dataset class does
        alleles_dict = {'v': sorted(list(dataconfig_library.get_allele_dict('v'))),
                        'd': sorted(list(dataconfig_library.get_allele_dict('d'))),
                        'j': sorted(list(dataconfig_library.get_allele_dict('j')))
                        }

        # Add Short D Label as Last Label
        alleles_dict['d'] += ['Short-D']

        for gene in ['v', 'd', 'j']:
            setattr(self, f"{gene}_allele_name_to_index", {name: index for index, name in enumerate(alleles_dict[gene])})
            setattr(self, f"{gene}_allele_index_to_name", {index: name for index, name in enumerate(alleles_dict[gene])})


    def get_mapping(self, key, allele_type):
        """This function takes either an allele and returns the index or vice versa."""
        if allele_type not in {'v', 'd', 'j'}:
            raise ValueError(f"Invalid allele type: {allele_type}. Must be one of 'v', 'd', 'j'.")

        # Select the appropriate mapping based on allele type and key type
        if isinstance(key, str):
            mapping = getattr(self, f"{allele_type}_allele_name_to_index")
            error_message = f"Allele {key} not found in {allele_type} alleles of the data config"
        elif isinstance(key, int):
            mapping = getattr(self, f"{allele_type}_allele_index_to_name")
            error_message = f"Index {key} not found in {allele_type} alleles of the data config"
        else:
            raise TypeError(f"Invalid key type: {type(key)}. Must be either str or int.")

        # Return the value or raise an error if the key is not found
        if key in mapping:
            return mapping[key]
        else:
            raise ValueError(error_message)

    def to_dict_form(self,likelihoods,allele_type):
        return {self.get_mapping(i,allele_type):likelihoods[i] for i in range(len(likelihoods))}

    def to_numpy(self,dict_form):
        """
        this function will take a list of dicationaries and return a numpy array of their values
        each dictionary is a row in the array.
        Args:
            dict_form:

        Returns:

        """
        return np.vstack([list(i.values()) for i in dict_form])
    def execute(self, predict_object: PredictObject):
        if predict_object.script_arguments.custom_genotype is None:
            self.log('No custom genotype provided, skipping likelihood adjustment.')
            return predict_object
        else:
            self.log("Adjusting Likelihood Given Genotype...")
            if predict_object.script_arguments.custom_genotype is not None:
                self.genotype_checker = IsInGenotype(predict_object.genotype)
            else:
                self.genotype_checker = IsInGenotype(predict_object.data_config_library)

            self.initialize_allele_index_translation_maps(predict_object.data_config_library)

            # iterate over the predicted likelihoods remove non-genotype outputs and redistribute the likelihoods
            for allele in ['v', 'd', 'j']:
                processed = []
                iterator = tqdm(predict_object.processed_predictions[allele + '_allele'], desc=f"Processing {allele} Allele Likelyhoods")
                for row in iterator:
                    dict_form = self.to_dict_form(row,allele)
                    processed.append(self.bounded_redistribution(dict_form))

                predict_object.processed_predictions[allele + '_allele'] = self.to_numpy(processed)

            return predict_object
