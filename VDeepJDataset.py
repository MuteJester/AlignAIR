import pandas as pd
from tqdm.auto import tqdm
from VDeepJDataPrepper import VDeepJDataPrepper
import numpy as np
import tensorflow as tf


class VDeepJDataset:
    def __init__(self, data_path, max_sequence_length=512, nrows=None):
        self.max_sequence_length = max_sequence_length
        self.data_prepper = VDeepJDataPrepper(self.max_sequence_length)
        self.data = pd.read_table(
            data_path,
            usecols=[
                "sequence",
                "v_call",
                "d_call",
                "j_call",
                "v_sequence_end",
                "d_sequence_start",
                "j_sequence_start",
                "j_sequence_end",
                "d_sequence_end",
                "v_sequence_start",
            ],
            nrows=nrows,
        )

        self.derive_call_dictionaries()
        self.derive_call_sections()
        self.derive_call_one_hot_representation()
        self.derive_counts()
        self.derive_sub_classes_dict()

    def derive_counts(self):
        self.v_family_count = len(self.v_family_call_ohe)
        self.v_gene_count = len(self.v_gene_call_ohe)
        self.v_allele_count = len(self.v_allele_call_ohe)

        self.d_family_count = len(self.d_family_call_ohe)
        self.d_gene_count = len(self.d_gene_call_ohe)
        self.d_allele_count = len(self.d_allele_call_ohe)

        self.j_gene_count = len(self.j_gene_call_ohe)
        self.j_allele_count = len(self.j_allele_call_ohe)

    def derive_call_one_hot_representation(self):
        (
            self.v_family_label_dict,
            self.v_family_call_ohe,
            self.v_family_call_ohe_np,
            self.v_gene_label_dict,
            self.v_gene_call_ohe,
            self.v_gene_call_ohe_np,
            self.v_allele_label_dict,
            self.v_allele_call_ohe,
            self.v_allele_call_ohe_np,
        ) = self.data_prepper.convert_calls_to_one_hot(
            self.v_gene, self.v_allele, self.v_family
        )

        (
            self.d_family_label_dict,
            self.d_family_call_ohe,
            self.d_family_call_ohe_np,
            self.d_gene_label_dict,
            self.d_gene_call_ohe,
            self.d_gene_call_ohe_np,
            self.d_allele_label_dict,
            self.d_allele_call_ohe,
            self.d_allele_call_ohe_np,
        ) = self.data_prepper.convert_calls_to_one_hot(
            self.d_gene, self.d_allele, self.d_family
        )

        (
            self.j_gene_label_dict,
            self.j_gene_call_ohe,
            self.j_gene_call_ohe_np,
            self.j_allele_label_dict,
            self.j_allele_call_ohe,
            self.j_allele_call_ohe_np,
        ) = self.data_prepper.convert_calls_to_one_hot(self.j_gene, self.j_allele)

        self.label_num_sub_classes_dict = {
            "V": {
                "family": self.v_family_label_dict,
                "gene": self.v_gene_label_dict,
                "allele": self.v_allele_label_dict,
            },
            "D": {
                "family": self.d_family_label_dict,
                "gene": self.d_gene_label_dict,
                "allele": self.d_allele_label_dict,
            },
            "J": {
                "gene": self.j_gene_label_dict,
                "allele": self.j_allele_label_dict,
            },
        }

    def derive_call_sections(self):
        self.v_family = [self.v_dict[x]["family"] for x in tqdm(self.data.v_call)]
        self.d_family = [self.d_dict[x]["family"] for x in tqdm(self.data.d_call)]

        self.v_gene = [self.v_dict[x]["gene"] for x in tqdm(self.data.v_call)]
        self.d_gene = [self.d_dict[x]["gene"] for x in tqdm(self.data.d_call)]
        self.j_gene = [self.j_dict[x]["gene"] for x in tqdm(self.data.j_call)]

        self.v_allele = [self.v_dict[x]["allele"] for x in tqdm(self.data.v_call)]
        self.d_allele = [self.d_dict[x]["allele"] for x in tqdm(self.data.d_call)]
        self.j_allele = [self.j_dict[x]["allele"] for x in tqdm(self.data.j_call)]

    def derive_call_dictionaries(self):
        (
            self.v_dict,
            self.d_dict,
            self.j_dict,
        ) = self.data_prepper.get_family_gene_allele_map(
            self.data.v_call.unique(),
            self.data.d_call.unique(),
            self.data.j_call.unique(),
        )

    def get_train_generator(self, batch_size, corrupt_beginning=False):
        return self.data_prepper.get_train_dataset(
            self.data,
            self.v_family_call_ohe_np,
            self.v_gene_call_ohe_np,
            self.v_allele_call_ohe_np,
            self.d_family_call_ohe_np,
            self.d_gene_call_ohe_np,
            self.d_allele_call_ohe_np,
            self.j_gene_call_ohe_np,
            self.j_allele_call_ohe_np,
            batch_size,
            train=True,
            corrupt_beginning=corrupt_beginning,
        )

    def get_eval_generator(self, batch_size):
        return self.data_prepper.get_train_dataset(
            self.data,
            self.v_family_call_ohe_np,
            self.v_gene_call_ohe_np,
            self.v_allele_call_ohe_np,
            self.d_family_call_ohe_np,
            self.d_gene_call_ohe_np,
            self.d_allele_call_ohe_np,
            self.j_gene_call_ohe_np,
            self.j_allele_call_ohe_np,
            batch_size,
            train=False,
            corrupt_beginning=False,
        )

    def generate_model_params(self):
        return {
            "max_seq_length": self.max_sequence_length,
            "v_family_count": self.v_family_count,
            "v_gene_count": self.v_gene_count,
            "v_allele_count": self.v_allele_count,
            "d_family_count": self.d_family_count,
            "d_gene_count": self.d_gene_count,
            "d_allele_count": self.d_allele_count,
            "j_gene_count": self.j_gene_count,
            "j_allele_count": self.j_allele_count,
            "ohe_sub_classes_dict": self.ohe_sub_classes_dict,
        }

    def create_vector(self, idxs, length):
        vector = np.zeros(length)
        vector[idxs] = 1
        return vector

    def derive_sub_classes_dict(self):
        self.label_name_sub_classes_dict = self.data_prepper.load_sub_classes_dict()

        self.ohe_sub_classes_dict = {
            "v": {"family": {}, "gene": {}},
            "d": {"family": {}, "gene": {}},
            "j": {"gene": {}},
        }
        counts_dict = {
            "v_family_count": self.v_family_count,
            "v_gene_count": self.v_gene_count,
            "v_allele_count": self.v_allele_count,
            "d_family_count": self.d_family_count,
            "d_gene_count": self.d_gene_count,
            "d_allele_count": self.d_allele_count,
            "j_gene_count": self.j_gene_count,
            "j_allele_count": self.j_allele_count,
        }
        for v_d_or_j in ["V", "D", "J"]:
            # Calculate the masked gene vectors for families
            if v_d_or_j in ["V", "D"]:
                for fam in self.label_name_sub_classes_dict[v_d_or_j].keys():
                    label_num = self.label_num_sub_classes_dict[v_d_or_j]["family"][fam]
                    wanted_keys = list(
                        self.label_name_sub_classes_dict[v_d_or_j][fam].keys()
                    )
                    possible_idxs = []
                    for wanted_key in wanted_keys:
                        possible_idxs.append(
                            self.label_num_sub_classes_dict[v_d_or_j]["gene"][
                                wanted_key
                            ]
                        )
                    self.ohe_sub_classes_dict[v_d_or_j]["family"][
                        label_num
                    ] = self.create_vector(
                        possible_idxs, counts_dict[v_d_or_j + "_gene_count"]
                    )
                # Calculate the masked allels vectors for genes
                for fam, fam_dict in self.label_name_sub_classes_dict[v_d_or_j].items():
                    fam_label_num = self.label_num_sub_classes_dict[v_d_or_j]["family"][
                        fam
                    ]
                    self.ohe_sub_classes_dict[v_d_or_j]["gene"][fam_label_num] = {}
                    for gene in fam_dict.keys():
                        if (
                            gene
                            not in self.ohe_sub_classes_dict[v_d_or_j]["gene"][
                                fam_label_num
                            ].keys()
                        ):
                            label_num = self.label_num_sub_classes_dict[v_d_or_j][
                                "gene"
                            ][gene]
                            wanted_keys = list(
                                self.label_name_sub_classes_dict[v_d_or_j][fam][
                                    gene
                                ].keys()
                            )
                            possible_idxs = []
                            for wanted_key in wanted_keys:
                                possible_idxs.append(
                                    self.label_num_sub_classes_dict[v_d_or_j]["allele"][
                                        wanted_key
                                    ]
                                )
                            self.ohe_sub_classes_dict[v_d_or_j]["gene"][fam_label_num][
                                label_num
                            ] = self.create_vector(
                                possible_idxs, counts_dict[v_d_or_j + "_allele_count"]
                            )
            else:  # J
                # Calculate the masked allels vectors for genes
                for gene in self.label_name_sub_classes_dict[v_d_or_j].keys():
                    if gene not in self.ohe_sub_classes_dict[v_d_or_j]["gene"].keys():
                        label_num = self.label_num_sub_classes_dict[v_d_or_j]["gene"][
                            gene
                        ]
                        wanted_keys = list(
                            self.label_name_sub_classes_dict[v_d_or_j][gene].keys()
                        )
                        possible_idxs = []
                        for wanted_key in wanted_keys:
                            possible_idxs.append(
                                self.label_num_sub_classes_dict[v_d_or_j]["allele"][
                                    wanted_key
                                ]
                            )
                        self.ohe_sub_classes_dict[v_d_or_j]["gene"][
                            label_num
                        ] = self.create_vector(
                            possible_idxs, counts_dict[v_d_or_j + "_allele_count"]
                        )

        ### Just for assertion, can be removed later ###
        v_family_keys = list(self.ohe_sub_classes_dict["V"]["family"].keys())
        v_family_keys.sort()
        assert v_family_keys == list(range(self.v_family_count))

        v_gene_keys = list(self.ohe_sub_classes_dict["V"]["gene"].keys())
        v_gene_keys.sort()
        assert v_gene_keys == list(range(self.v_family_count))

        d_family_keys = list(self.ohe_sub_classes_dict["D"]["family"].keys())
        d_family_keys.sort()
        assert d_family_keys == list(range(self.d_family_count))

        d_gene_keys = list(self.ohe_sub_classes_dict["D"]["gene"].keys())
        d_gene_keys.sort()
        assert d_gene_keys == list(range(self.d_family_count))

        j_gene_keys = list(self.ohe_sub_classes_dict["J"]["gene"].keys())
        j_gene_keys.sort()
        assert j_gene_keys == list(range(self.j_gene_count))
        ###

        # Stack all vectors (by index order) for useful indexing
        # V
        self.ohe_sub_classes_dict["V"]["family"] = np.stack(
            [
                self.ohe_sub_classes_dict["V"]["family"][i]
                for i in list(range(self.v_family_count))
            ]
        )
        self.ohe_sub_classes_dict["V"]["family"] = tf.convert_to_tensor(
            self.ohe_sub_classes_dict["V"]["family"], dtype=tf.float32
        )

        tmp_tensor = np.zeros(
            (self.v_family_count, self.v_gene_count, self.v_allele_count)
        )
        for fam_label_num in self.ohe_sub_classes_dict["V"]["gene"].keys():
            for gene_label_num in self.ohe_sub_classes_dict["V"]["gene"][
                fam_label_num
            ].keys():
                tmp_tensor[fam_label_num, gene_label_num] = self.ohe_sub_classes_dict[
                    "V"
                ]["gene"][fam_label_num][gene_label_num]
        tmp_tensor = tf.convert_to_tensor(tmp_tensor, dtype=tf.float32)
        self.ohe_sub_classes_dict["V"]["gene"] = tmp_tensor

        # D
        self.ohe_sub_classes_dict["D"]["family"] = np.stack(
            [
                self.ohe_sub_classes_dict["D"]["family"][i]
                for i in list(range(self.d_family_count))
            ]
        )
        self.ohe_sub_classes_dict["D"]["family"] = tf.convert_to_tensor(
            self.ohe_sub_classes_dict["D"]["family"], dtype=tf.float32
        )

        tmp_tensor = np.zeros(
            (self.d_family_count, self.d_gene_count, self.d_allele_count)
        )
        for fam_label_num in self.ohe_sub_classes_dict["D"]["gene"].keys():
            for gene_label_num in self.ohe_sub_classes_dict["D"]["gene"][
                fam_label_num
            ].keys():
                tmp_tensor[fam_label_num, gene_label_num] = self.ohe_sub_classes_dict[
                    "D"
                ]["gene"][fam_label_num][gene_label_num]
        tmp_tensor = tf.convert_to_tensor(tmp_tensor, dtype=tf.float32)
        self.ohe_sub_classes_dict["D"]["gene"] = tmp_tensor

        # J
        self.ohe_sub_classes_dict["J"]["gene"] = np.stack(
            [
                self.ohe_sub_classes_dict["J"]["gene"][i]
                for i in list(range(self.j_gene_count))
            ]
        )
        self.ohe_sub_classes_dict["J"]["gene"] = tf.convert_to_tensor(
            self.ohe_sub_classes_dict["J"]["gene"], dtype=tf.float32
        )

    def __len__(self):
        return len(self.data)
