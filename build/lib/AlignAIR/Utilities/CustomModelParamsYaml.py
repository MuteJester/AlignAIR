import yaml

from AlignAIR.Utilities.step_utilities import DataConfigLibrary


class CustomModelParamsYaml:
    accepted_keys = ['v_allele_latent_size', 'd_allele_latent_size', 'j_allele_latent_size',
                     'v_allele_count', 'd_allele_count', 'j_allele_count']
    def __init__(self, yaml_path):
        """
        Initialize the parser with the path to the YAML file.
        Reads the YAML file, validates keys, and populates instance variables.
        """
        self.yaml_path = yaml_path
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
                    if key in self.accepted_keys:
                        setattr(self, key, data[key])
                    else:
                        raise ValueError(f"Key '{key}' is not allowed in the YAML file. must be one of {','.join(self.accepted_keys)}")

        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file '{self.yaml_path}' not found.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")


    def __repr__(self):
        """
        String representation of the class showing the lists for `v`, `d`, and `j`.
        """
        rep_string = ""
        for key in self.accepted_keys:
            if hasattr(self, key):
                rep_string += f"{key}: {getattr(self, key)}\n"
        return rep_string


