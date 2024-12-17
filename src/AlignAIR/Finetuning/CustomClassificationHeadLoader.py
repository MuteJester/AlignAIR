import logging
import tensorflow as tf


class CustomClassificationHeadLoader:
    """
    CustomClassificationHeadLoader class is responsible for loading a pretrained model's weights
    and applying custom classification heads with user-defined sizes. It freezes all the weights
    from the pretrained model except for the new classification heads.
    """

    def __init__(self, model_class, pretrained_path,
                 max_seq_length,
                 pretrained_v_allele_head_size,
                 pretrained_d_allele_head_size,
                 pretrained_j_allele_head_size,
                 custom_v_allele_head_size='same',
                 custom_d_allele_head_size='same',
                 custom_j_allele_head_size='same'):

        self.max_seq_length = max_seq_length
        self.input_shape = (max_seq_length, 1)
        self.custom_v_allele_head_size = custom_v_allele_head_size
        self.custom_d_allele_head_size = custom_d_allele_head_size
        self.custom_j_allele_head_size = custom_j_allele_head_size
        self.pretrained_v_allele_head_size = pretrained_v_allele_head_size
        self.pretrained_d_allele_head_size = pretrained_d_allele_head_size
        self.pretrained_j_allele_head_size = pretrained_j_allele_head_size

        self.pretrained_model = None
        self.pretrained_model_params = {
            'max_seq_length':max_seq_length,
            'v_allele_count':self.pretrained_v_allele_head_size,
            'd_allele_count':self.pretrained_d_allele_head_size,
            'j_allele_count':self.pretrained_j_allele_head_size,
        }

        self.custom_model_params = {
            'max_seq_length':max_seq_length,
            'v_allele_count':self.custom_v_allele_head_size if self.custom_v_allele_head_size != 'same' else self.pretrained_v_allele_head_size,
            'd_allele_count':self.custom_d_allele_head_size if self.custom_d_allele_head_size != 'same' else self.pretrained_d_allele_head_size,
            'j_allele_count':self.custom_j_allele_head_size if self.custom_j_allele_head_size != 'same' else self.pretrained_j_allele_head_size,
            'v_allele_latent_size':self.custom_v_allele_head_size, #multiply by latent_size_factor after model is loaded
            'd_allele_latent_size':self.custom_d_allele_head_size, #multiply by latent_size_factor after model is loaded
            'j_allele_latent_size':self.custom_j_allele_head_size, #multiply by latent_size_factor after model is loaded

        }


        self.model_class = model_class
        self.model = None
        self.pretrained_path = pretrained_path
        self.load_model(self.pretrained_path)

    def copy_weights(self, pretrained_model, custom_model):
        mismatched_layers = set()  # To store the names of layers with mismatched weights
        mismatched_indexes = []  # To store the mismatched weight indices

        # Get the weights from the original model
        original_weights = pretrained_model.get_weights()
        modified_weights = custom_model.get_weights()

        for i, (ow, mw) in enumerate(zip(original_weights, modified_weights)):
            if ow.shape != mw.shape:
                print(
                    f"Mismatch at index {i}: Original {ow.shape}, Modified {mw.shape}, name: {pretrained_model.weights[i].name}")
                # Add the layer name associated with this weight
                weight_name = pretrained_model.weights[i].name
                for layer in pretrained_model.layers:
                    if any(w.name == weight_name for w in layer.weights):
                        mismatched_layers.add(layer.name)
                        break
                # Record the mismatched index
                mismatched_indexes.append(i)
                print('______________________________')

        # Step 2: Copy weights while skipping mismatched layers
        for original_layer, modified_layer in zip(pretrained_model.layers, custom_model.layers):
            # Skip mismatched layers
            if original_layer.name in mismatched_layers:
                print(f"Skipping layer {original_layer.name} due to mismatched weights.")
                continue

            # Copy weights and freeze layer
            if original_layer.get_weights():  # Ensure the layer has weights to copy
                modified_layer.set_weights(original_layer.get_weights())
                modified_layer.trainable = False  # Freeze layer
                # print(f"Copied and froze weights for layer {original_layer.name}")


    def load_model(self, pretrained_path):
        """Loads the pretrained model weights onto the custom model settings."""
        self.pretrained_model = self.model_class(**self.pretrained_model_params)
        self.pretrained_model.build({'tokenized_sequence': self.input_shape})
        self.pretrained_model.load_weights(pretrained_path)


        self.custom_model_params['v_allele_latent_size'] = self.pretrained_v_allele_head_size * self.pretrained_model.latent_size_factor
        self.custom_model_params['d_allele_latent_size'] = self.pretrained_d_allele_head_size * self.pretrained_model.latent_size_factor
        self.custom_model_params['j_allele_latent_size'] = self.pretrained_j_allele_head_size * self.pretrained_model.latent_size_factor

        self.model = self.model_class(**self.custom_model_params)
        self.model.build({'tokenized_sequence': self.input_shape})

        self.copy_weights(self.pretrained_model, self.model)

        logging.info(f"Pretrained model weights loaded from {pretrained_path}.")

    def save_model_weight(self, path):
        """Saves the model weights to the specified path."""
        self.model.save_weights(path)
        logging.info(f"Model weights saved to {path}.")
    def save_model_h5(self):
        """Saves the model to the specified path."""
        self.model.save("model.h5")
        logging.info(f"Model saved to model.h5.")