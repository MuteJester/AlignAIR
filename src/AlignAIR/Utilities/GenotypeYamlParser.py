import yaml
from GenAIRR.dataconfig import DataConfig


class GenotypeYamlParser:
    def __init__(self, yaml_path):
        """
        Initialize the parser with the path to the YAML file.
        Reads the YAML file, validates keys, and populates instance variables.
        """
        self.yaml_path = yaml_path
        self.v = []
        self.d = []
        self.j = []
        self._parse_yaml()

    def _parse_yaml(self):
        """
        Reads the YAML file and populates `v`, `d`, and `j` attributes.
        Ensures only keys `v`, `d`, and `j` are allowed.
        """
        try:
            with open(self.yaml_path, 'r') as file:
                data = yaml.safe_load(file)

                # Validate and populate keys
                if not isinstance(data, dict):
                    raise ValueError("YAML content must be a dictionary.")

                for key in data:
                    if key not in ['v', 'd', 'j']:
                        raise ValueError(f"Invalid key '{key}' in YAML file. Allowed keys are 'v', 'd', and 'j'.")

                # Populate attributes, default to empty lists for missing keys
                self.v = data.get('v', [])
                self.d = data.get('d', [])
                self.j = data.get('j', [])

                if 'Short-D' not in self.d:
                    self.d += ['Short-D']

                # Ensure the values are lists
                if not all([isinstance(self.v, list) , isinstance(self.d, list) , isinstance(self.j, list)]):
                    raise ValueError("Values for keys 'v', 'd', and 'j' must be lists.")

        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file '{self.yaml_path}' not found.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def test_intersection_with_data_config(self, dataconfig:DataConfig):
        """
        Test if the genotype alleles intersect with the data config alleles.
        this will raise an error if there are any alleles in the provided genotype that are not in the data config.
        This is due to the fact that the model was trained on a specific set of alleles and the genotype should be
        a subset of those alleles.
        """

        for gene in  ['v','d','j']:
            dataconfig_allele_names = dataconfig.allele_list(gene)
            dataconfig_allele_names = set([i.name for i in dataconfig_allele_names])
            current_alleles = set(getattr(self,gene))
            if 'Short-D' in dataconfig_allele_names:
                dataconfig_allele_names.remove('Short-D')
            if 'Short-D' in current_alleles:
                current_alleles.remove('Short-D')
            if (len(dataconfig_allele_names) > 0) and (len(current_alleles - dataconfig_allele_names) > 0):
                raise ValueError(f"Alleles in the genotype yaml file for {gene} are not present in the data config file")


    def __repr__(self):
        """
        String representation of the class showing the lists for `v`, `d`, and `j`.
        """
        return f"GenotypeLists(v={self.v}, d={self.d}, j={self.j})"



