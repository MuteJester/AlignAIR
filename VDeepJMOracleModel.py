import tensorflow.keras.backend as K
from tensorflow.keras.layers import Attention
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, concatenate, Input, Embedding, Dropout
import tensorflow as tf
from tensorflow.keras.constraints import unit_norm
from VDeepJLayers import CutoutLayer, ExtractGeneMask, Conv2D_and_BatchNorm, mod3_mse_regularization, \
    Conv1D_and_BatchNorm, ExtractGeneMask1D, TokenAndPositionEmbedding, MutationOracleBody
from tensorflow.keras import regularizers
from enum import Enum, auto
from VDeepJModel import ModelComponents


class VDeepJMOracleAllign(tf.keras.Model):

    def __init__(self, max_seq_length, v_family_count, v_gene_count, v_allele_count,
                 d_family_count, d_gene_count, d_allele_count, j_gene_count, j_allele_count):
        super(VDeepJMOracleAllign, self).__init__()
        # Model Params
        self.max_seq_length = int(max_seq_length)
        self.v_family_count, self.v_gene_count, self.v_allele_count = v_family_count, v_gene_count, v_allele_count
        self.d_family_count, self.d_gene_count, self.d_allele_count = d_family_count, d_gene_count, d_allele_count
        self.j_gene_count, self.j_allele_count = j_gene_count, j_allele_count
        self.v_class_weight, self.d_class_weight, self.j_class_weight = 0.5, 0.5, 0.5
        self.regression_weight, self.classification_weight, self.intersection_weight,self.mutation_oracle_weight\
            = 0.5, 0.5, 0.5, 0.5

        # Hyperparams + Constants
        self.regression_keys = ['v_start', 'v_end', 'd_start', 'd_end', 'j_start', 'j_end']
        self.classification_keys = ['v_family', 'v_gene', 'v_allele', 'd_family', 'd_gene', 'd_allele', 'j_gene',
                                    'j_allele']
        self.latent_size_factor = 2
        self.classification_middle_layer_activation = 'swish'

        # Tracking
        self.init_loss_tracking_variables()

        # Init Input Layers
        self._init_input_layers()

        # Init layers that Encode the Initial 4 RAW A-T-G-C Signals
        self._init_raw_signals_encoding_layers()
        self.initial_embedding_attention = Attention()

        # Init Mutation Oracle
        self.mutation_oracle = MutationOracleBody(activation='relu',latent_dim = 1024,name='mutation_oracle_body')
        self.mutation_oracle_head = Dense(activation='sigmoid',units=self.max_seq_length,name='mutation_oracle_head')

        # Init V/D/J Masked Input Signal Encoding Layers
        self._init_masked_v_signals_encoding_layers()
        self._init_masked_d_signals_encoding_layers()
        self._init_masked_j_signals_encoding_layers()

        self.concatenate_input = concatenate
        self.concatenated_input_embedding = TokenAndPositionEmbedding(vocab_size=6, emded_dim=32,
                                                                      maxlen=self.max_seq_length)  # Embedding(6, 32, input_length=int(max_seq_length))
        self.initial_feature_map_dropout = Dropout(0.3)

        self.concatenated_v_mask_input_embedding = TokenAndPositionEmbedding(vocab_size=6, emded_dim=32,
                                                                             maxlen=self.max_seq_length)  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_d_mask_input_embedding = TokenAndPositionEmbedding(vocab_size=6, emded_dim=32,
                                                                             maxlen=self.max_seq_length)  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)
        self.concatenated_j_mask_input_embedding = TokenAndPositionEmbedding(vocab_size=6, emded_dim=32,
                                                                             maxlen=self.max_seq_length)  # Embedding(6, 32, input_length=int(max_seq_length))  # (concatenated)

        # Init Interval Regression Related Layers
        self._init_interval_regression_layers()

        self.v_call_mask = CutoutLayer(max_seq_length, 'V', name='V_extract')
        self.d_call_mask = CutoutLayer(max_seq_length, 'D', name='D_extract')
        self.j_call_mask = CutoutLayer(max_seq_length, 'J', name='J_extract')

        self.v_mask_extractor = ExtractGeneMask1D()
        self.d_mask_extractor = ExtractGeneMask1D()
        self.j_mask_extractor = ExtractGeneMask1D()

        self.v_mask_attention = Attention(name='v_mask_mutation_attention')
        self.d_mask_attention = Attention(name='d_mask_mutation_attention')
        self.j_mask_attention = Attention(name='j_mask_mutation_attention')


        #  =========== V HEADS ======================
        # Init V Classification Related Layers
        self._init_v_classification_layers()
        # =========== D HEADS ======================
        # Init D Classification Related Layers
        self._init_d_classification_layers()
        # =========== J HEADS ======================
        # Init J Classification Related Layers
        self._init_j_classification_layers()

    def init_loss_tracking_variables(self):
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.insec_loss_tracker = tf.keras.metrics.Mean(name="insec_loss")
        self.mod3_mse_loss_tracker = tf.keras.metrics.Mean(name="mod3_mse_loss")
        self.total_ce_loss_tracker = tf.keras.metrics.Mean(name="total_classification_loss")
        self.mutation_oracle_loss_tracker = tf.keras.metrics.Mean(name="mutation_oracle")

    def reshape_and_cast_input(self, input_s):
        a = K.reshape(input_s, (-1, self.max_seq_length))
        a = K.cast(a, 'float32')
        return a

    def _init_input_layers(self):
        self.input_init = Input((self.max_seq_length, 1), name='seq_init')
        self.input_for_masked = Input((self.max_seq_length, 1), name='seq_masked')

    def _init_raw_signals_encoding_layers(self):
        self.conv_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_layer_4 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=3)

    def _init_masked_v_signals_encoding_layers(self):
        self.conv_v_layer_1 = Conv1D_and_BatchNorm(filters=128, kernel=3, max_pool=2)
        self.conv_v_layer_2 = Conv1D_and_BatchNorm(filters=256, kernel=3, max_pool=2)
        self.conv_v_layer_3 = Conv1D_and_BatchNorm(filters=512, kernel=3, max_pool=2)
        self.conv_v_layer_4 = Conv1D_and_BatchNorm(filters=128, kernel=2, max_pool=2)

    def _init_masked_d_signals_encoding_layers(self):
        self.conv_d_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_d_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_d_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_d_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_masked_j_signals_encoding_layers(self):
        self.conv_j_layer_1 = Conv1D_and_BatchNorm(filters=16, kernel=3, max_pool=2)
        self.conv_j_layer_2 = Conv1D_and_BatchNorm(filters=32, kernel=3, max_pool=2)
        self.conv_j_layer_3 = Conv1D_and_BatchNorm(filters=64, kernel=3, max_pool=2)
        self.conv_j_layer_4 = Conv1D_and_BatchNorm(filters=64, kernel=2, max_pool=2)

    def _init_j_classification_layers(self):
        self.j_gene_call_middle = Dense(self.j_gene_count * self.latent_size_factor,
                                        activation=self.classification_middle_layer_activation,
                                        name='j_gene_middle', kernel_regularizer=regularizers.l2(0.01))
        self.j_gene_call_head = Dense(self.j_gene_count, activation='softmax', name='j_gene')  # (v_feature_map)

        self.j_allele_call_middle = Dense(self.j_allele_count * self.latent_size_factor,
                                          activation=self.classification_middle_layer_activation,
                                          name='j_allele_middle', kernel_regularizer=regularizers.l2(0.01))
        self.j_allele_call_head = Dense(self.j_allele_count, activation='softmax',
                                        name='j_allele')  # (v_feature_map)

        self.j_gene_call_gene_allele_concat = concatenate

    def _init_d_classification_layers(self):
        self.d_family_call_middle = Dense(self.d_family_count * self.latent_size_factor,
                                          activation=self.classification_middle_layer_activation,
                                          name='d_family_middle', kernel_regularizer=regularizers.l2(0.01))
        self.d_family_call_head = Dense(self.d_family_count, activation='softmax',
                                        name='d_family')  # (v_feature_map)

        self.d_gene_call_middle = Dense(self.d_gene_count * self.latent_size_factor,
                                        activation=self.classification_middle_layer_activation,
                                        name='d_gene_middle', kernel_regularizer=regularizers.l2(0.01))
        self.d_gene_call_head = Dense(self.d_gene_count, activation='softmax', name='d_gene')  # (v_feature_map)

        self.d_allele_call_middle = Dense(self.d_allele_count * self.latent_size_factor,
                                          activation=self.classification_middle_layer_activation,
                                          name='d_allele_middle', kernel_regularizer=regularizers.l2(0.01))
        self.d_allele_call_head = Dense(self.d_allele_count, activation='softmax',
                                        name='d_allele')  # (v_feature_map)

        self.d_gene_call_family_gene_concat = concatenate
        self.d_gene_call_gene_allele_concat = concatenate

    def _init_v_classification_layers(self):

        self.v_family_call_middle = Dense(self.v_family_count * self.latent_size_factor,
                                          activation=self.classification_middle_layer_activation,
                                          name='v_family_middle', kernel_regularizer=regularizers.l2(0.03))

        self.v_family_call_head = Dense(self.v_family_count, activation='softmax',
                                        name='v_family')  # (v_feature_map)

        self.v_family_dropout = Dropout(0.2)

        self.v_gene_call_middle = Dense(self.v_gene_count * self.latent_size_factor,
                                        activation=self.classification_middle_layer_activation,
                                        name='v_gene_middle', kernel_regularizer=regularizers.l2(0.03))
        self.v_gene_call_head = Dense(self.v_gene_count, activation='softmax', name='v_gene')  # (v_feature_map)
        self.v_gene_dropout = Dropout(0.2)

        self.v_allele_call_middle = Dense(self.v_allele_count * self.latent_size_factor,
                                          activation=self.classification_middle_layer_activation,
                                          name='v_allele_middle', kernel_regularizer=regularizers.l2(0.03))
        self.v_allele_call_head = Dense(self.v_allele_count, activation='softmax',
                                        name='v_allele')  # (v_feature_map)
        self.v_allele_dropout = Dropout(0.2)
        self.v_allele_feature_distill = Dense(self.v_family_count + self.v_gene_count + self.v_allele_count,
                                              activation=self.classification_middle_layer_activation,
                                              name='v_gene_allele_distill', kernel_regularizer=regularizers.l2(0.03))



        self.v_gene_call_family_gene_concat = concatenate
        self.v_gene_call_gene_allele_concat = concatenate

    def _init_interval_regression_layers(self):
        # act = tf.keras.layers.LeakyReLU()
        act = tf.keras.activations.swish
        self.v_start_mid = Dense(32, activation=act, kernel_constraint=unit_norm())  # (concatenated_path)
        self.v_start_out = Dense(1, activation='relu', name='v_start')  # (v_end_mid)

        self.v_end_mid = Dense(32, activation=act, kernel_constraint=unit_norm())  # (concatenated_path)
        self.v_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.v_end_out = Dense(1, activation='relu', name='v_end')  # (v_end_mid)

        self.d_start_mid = Dense(32, activation=act, kernel_constraint=unit_norm())  # (concatenated_path)
        self.d_start_out = Dense(1, activation='relu', name='d_start')  # (d_start_mid)

        self.d_end_mid = Dense(32, activation=act, kernel_constraint=unit_norm())  # (concatenated_path)
        self.d_end_mid_concat = concatenate  # ([d_end_mid,d_start_mid])
        self.d_end_out = Dense(1, activation='relu', name='d_end')  # (d_end_mid)

        self.j_start_mid = Dense(32, activation=act, kernel_constraint=unit_norm())  # (concatenated_path)
        self.j_start_out = Dense(1, activation='relu', name='j_start')  # (j_start_mid)

        self.j_end_mid = Dense(32, activation=act, kernel_constraint=unit_norm())  # (concatenated_path)
        self.j_end_mid_concat = concatenate  # ([j_end_mid,j_start_mid])
        self.j_end_out = Dense(1, activation='relu', name='j_end')  # (j_end_mid)

    def _encode_features(self, input, layer):
        a = input
        a = self.reshape_and_cast_input(a)
        return layer(a)

    def _predict_intervals(self, concatenated_signals):
        v_start_middle = self.v_start_mid(concatenated_signals)
        v_start = self.v_start_out(v_start_middle)

        v_end_middle = self.v_end_mid(concatenated_signals)
        v_end_middle = self.v_end_mid_concat([v_end_middle, v_start_middle])
        # This is the predicted index where the V Gene ends
        v_end = self.v_end_out(v_end_middle)

        # Middle layer for D start prediction
        d_start_middle = self.d_start_mid(concatenated_signals)
        # This is the predicted index where the D Gene starts
        d_start = self.d_start_out(d_start_middle)

        d_end_middle = self.d_end_mid(concatenated_signals)
        d_end_middle = self.d_end_mid_concat([d_end_middle, d_start_middle])
        # This is the predicted index where the D Gene ends
        d_end = self.d_end_out(d_end_middle)

        j_start_middle = self.j_start_mid(concatenated_signals)
        # This is the predicted index where the J Gene starts
        j_start = self.j_start_out(j_start_middle)

        j_end_middle = self.j_end_mid(concatenated_signals)
        j_end_middle = self.j_end_mid_concat([j_end_middle, j_start_middle])
        # This is the predicted index where the J Gene ends
        j_end = self.j_end_out(j_end_middle)
        return v_start, v_end, d_start, d_end, j_start, j_end

    def _predict_vdj_set(self, v_feature_map, d_feature_map, j_feature_map):
        # ============================ V =============================
        v_family_middle = self.v_family_call_middle(v_feature_map)
        v_family_middle = self.v_family_dropout(v_family_middle)
        v_family = self.v_family_call_head(v_family_middle)

        v_gene_middle = self.v_gene_call_middle(v_feature_map)
        v_gene_middle = self.v_gene_call_family_gene_concat([v_gene_middle, v_family_middle])
        v_gene_middle = self.v_gene_dropout(v_gene_middle)
        v_gene = self.v_gene_call_head(v_gene_middle)

        v_allele_middle = self.v_allele_call_middle(v_feature_map)
        v_allele_middle = self.v_gene_call_gene_allele_concat([v_family_middle, v_gene_middle, v_allele_middle])
        v_allele_middle = self.v_allele_dropout(v_allele_middle)
        v_allele_middle = self.v_allele_feature_distill(v_allele_middle)
        v_allele = self.v_allele_call_head(v_allele_middle)
        # ============================ D =============================
        d_family_middle = self.d_family_call_middle(d_feature_map)
        d_family = self.d_family_call_head(d_family_middle)

        d_gene_middle = self.d_gene_call_middle(d_feature_map)
        d_gene_middle = self.d_gene_call_family_gene_concat([d_gene_middle, d_family_middle])
        d_gene = self.d_gene_call_head(d_gene_middle)

        d_allele_middle = self.d_allele_call_middle(d_feature_map)
        d_allele_middle = self.d_gene_call_gene_allele_concat([d_allele_middle, d_gene_middle])
        d_allele = self.d_allele_call_head(d_allele_middle)
        # ============================ J =============================
        j_gene_middle = self.j_gene_call_middle(j_feature_map)
        j_gene = self.j_gene_call_head(j_gene_middle)

        j_allele_middle = self.j_allele_call_middle(j_feature_map)
        j_allele_middle = self.j_gene_call_gene_allele_concat([j_allele_middle, j_gene_middle])
        j_allele = self.j_allele_call_head(j_allele_middle)

        return v_family, v_gene, v_allele, d_family, d_gene, d_allele, j_gene, j_allele

    def _encode_masked_v_signal(self, concatenated_v_mask_input_embedding):
        v_conv_layer_1 = self.conv_v_layer_1(concatenated_v_mask_input_embedding)
        v_conv_layer_2 = self.conv_v_layer_2(v_conv_layer_1)
        v_conv_layer_3 = self.conv_v_layer_3(v_conv_layer_2)
        v_feature_map = self.conv_v_layer_4(v_conv_layer_3)
        v_feature_map = Flatten()(v_feature_map)
        return v_feature_map

    def _encode_masked_d_signal(self, concatenated_d_mask_input_embedding):
        d_conv_layer_1 = self.conv_d_layer_1(concatenated_d_mask_input_embedding)
        d_conv_layer_2 = self.conv_d_layer_2(d_conv_layer_1)
        d_conv_layer_3 = self.conv_d_layer_3(d_conv_layer_2)
        d_feature_map = self.conv_d_layer_4(d_conv_layer_3)
        d_feature_map = Flatten()(d_feature_map)
        return d_feature_map

    def _encode_masked_j_signal(self, concatenated_j_mask_input_embedding):
        j_conv_layer_1 = self.conv_j_layer_1(concatenated_j_mask_input_embedding)
        j_conv_layer_2 = self.conv_j_layer_2(j_conv_layer_1)
        j_conv_layer_3 = self.conv_j_layer_3(j_conv_layer_2)
        j_feature_map = self.conv_j_layer_4(j_conv_layer_3)
        j_feature_map = Flatten()(j_feature_map)
        return j_feature_map

    def call(self, inputs):

        # STEP 1 : Produce embeddings for the input sequence
        input_seq = self.reshape_and_cast_input(inputs['tokenized_sequence'])
        concatenated_input_embedding = self.concatenated_input_embedding(input_seq)
        concatenated_input_embedding = self.initial_embedding_attention(
                [concatenated_input_embedding, concatenated_input_embedding])



        # STEP 2: Run Embedded sequence through 1D convolution to distill temporal features
        conv_layer_1 = self.conv_layer_1(concatenated_input_embedding)
        conv_layer_2 = self.conv_layer_2(conv_layer_1)
        conv_layer_3 = self.conv_layer_3(conv_layer_2)
        last_conv_layer = self.conv_layer_4(conv_layer_3)

        # Step 2.5 Send Embeddings to Mutation Oracle in Parallel
        mutation_oracle_latnet = self.mutation_oracle(concatenated_input_embedding)
        mutation_oracle_head = self.mutation_oracle_head(mutation_oracle_latnet)

        # STEP 3 : Flatten The Feature Derived from the 1D conv layers
        concatenated_signals = last_conv_layer
        concatenated_signals = Flatten()(concatenated_signals)
        concatenated_signals = self.initial_feature_map_dropout(concatenated_signals)

        # STEP 4 : Predict The Intervals That Contain The V,D and J Genes using (V_start,V_end,D_Start,D_End,J_Start,J_End)
        v_start, v_end, d_start, d_end, j_start, j_end = self._predict_intervals(concatenated_signals)

        # STEP 5: Use predicted masks to create a binary vector with the appropriate intervals to  "cutout" the relevant V,D and J section from the input
        v_mask = self.v_call_mask([v_start, v_end])
        d_mask = self.d_call_mask([d_start, d_end])
        j_mask = self.j_call_mask([j_start, j_end])

        # Get the second copy of the inputs
        input_seq_for_masked = self.reshape_and_cast_input(inputs['tokenized_sequence_for_masking'])

        # STEP 5: Multiply the mask with the input vector to turn of (set as zero) all position that dont match mask interval
        masked_sequence_v = self.v_mask_extractor(
            (input_seq_for_masked, v_mask))
        masked_sequence_d = self.d_mask_extractor(
            (input_seq_for_masked, d_mask))
        masked_sequence_j = self.j_mask_extractor(
            (input_seq_for_masked, j_mask))

        # apply attention on masked sequences
        masked_sequence_v = self.v_mask_attention([masked_sequence_v,mutation_oracle_head])
        masked_sequence_d = self.d_mask_attention([masked_sequence_d,mutation_oracle_head])
        masked_sequence_j = self.j_mask_attention([masked_sequence_j,mutation_oracle_head])

        # STEP 6: Extract new Feature
        # Create Embeddings from the New 4 Channel Concatenated Signal using an Embeddings Layer - Apply for each Gene
        v_mask_input_embedding = self.concatenated_v_mask_input_embedding(masked_sequence_v)
        d_mask_input_embedding = self.concatenated_d_mask_input_embedding(masked_sequence_d)
        j_mask_input_embedding = self.concatenated_j_mask_input_embedding(masked_sequence_j)

        # Pass The Embeddings Generated Above Thorough 2D Convolutional Feature Extractor Layer
        v_feature_map = self._encode_masked_v_signal(v_mask_input_embedding)
        d_feature_map = self._encode_masked_d_signal(d_mask_input_embedding)
        j_feature_map = self._encode_masked_j_signal(j_mask_input_embedding)

        # STEP 8: Predict The V,D and J genes
        v_family, v_gene, v_allele, d_family, d_gene, d_allele, j_gene, j_allele = self._predict_vdj_set(v_feature_map,
                                                                                                         d_feature_map,
                                                                                                         j_feature_map)

        return {'mutations':mutation_oracle_head,'v_start': v_start, 'v_end': v_end, 'd_start': d_start, 'd_end': d_end, 'j_start': j_start,
                'j_end': j_end,
                'v_family': v_family, 'v_gene': v_gene, 'v_allele': v_allele,
                'd_family': d_family, 'd_gene': d_gene, 'd_allele': d_allele,
                'j_gene': j_gene, 'j_allele': j_allele}

    def c2f32(self, x):
        # cast keras tensor to float 32
        return K.cast(x, 'float32')

    def call_hierarchy_loss(self, family_true, gene_true, allele_true, family_pred, gene_pred, allele_pred):
        if family_true != None:
            family_loss = (
                K.categorical_crossentropy(family_true, family_pred))  # K.categorical_crossentropy
        gene_loss = (K.categorical_crossentropy(gene_true, gene_pred))
        allele_loss = (K.categorical_crossentropy(allele_true, allele_pred))

        # family_loss_mean = K.mean(family_loss)
        # gene_loss_mean = K.mean(gene_loss)
        # allele_loss_mean = K.mean(allele_loss)

        # Penalty for wrong family classification
        penalty_upper = K.constant([10.0])
        penalty_mid = K.constant([5.0])
        penalty_lower = K.constant([1.0])

        if family_true != None:
            family_penalty = K.switch(K.not_equal(K.argmax(family_true), K.argmax(family_pred)), penalty_upper,
                                      penalty_lower)
            gene_penalty = K.switch(K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)), penalty_mid, penalty_lower)
        else:
            family_penalty = K.switch(K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)), penalty_upper,
                                      penalty_lower)

        # Compute the final loss based on the constraint
        if family_true != None:
            loss = K.switch(K.not_equal(K.argmax(family_true), K.argmax(family_pred)),

                            family_penalty * (family_loss + gene_loss + allele_loss),

                            K.switch(K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),

                                     family_loss + gene_penalty * (gene_loss + allele_loss),
                                     family_loss + gene_loss + penalty_upper * allele_loss)
                            )
        else:
            loss = K.switch(K.not_equal(K.argmax(gene_true), K.argmax(gene_pred)),

                            family_penalty * (gene_loss + allele_loss),

                            gene_loss + penalty_upper * allele_loss
                            )

        return K.mean(loss)

    def multi_task_loss_v2(self, y_true, y_pred):
        # Extract the regression and classification outputs
        regression_true = [self.c2f32(y_true[k]) for k in self.regression_keys]
        regression_pred = [self.c2f32(y_pred[k]) for k in self.regression_keys]
        classification_true = [self.c2f32(y_true[k]) for k in self.classification_keys]
        classification_pred = [self.c2f32(y_pred[k]) for k in self.classification_keys]
        mutations_true,mutations_pred = y_true['mutations'],y_pred['mutations']

        v_start, v_end, d_start, d_end, j_start, j_end = regression_pred
        # ========================================================================================================================

        # Compute the intersection loss
        v_intersection_loss = K.maximum(0.0, K.minimum(v_end, d_end) - K.maximum(v_start, d_start)) + K.maximum(0.0,
                                                                                                                K.minimum(
                                                                                                                    v_end,
                                                                                                                    j_end) - K.maximum(
                                                                                                                    v_start,
                                                                                                                    j_start))
        d_intersection_loss = K.maximum(0.0, K.minimum(d_end, j_end) - K.maximum(d_start, j_start)) + K.maximum(0.0,
                                                                                                                K.minimum(
                                                                                                                    d_end,
                                                                                                                    v_end) - K.maximum(
                                                                                                                    d_start,
                                                                                                                    v_start))
        j_intersection_loss = K.maximum(0.0, K.minimum(j_end, self.max_seq_length) - K.maximum(j_start, j_end))
        total_intersection_loss = v_intersection_loss + d_intersection_loss + j_intersection_loss
        # ========================================================================================================================

        # Compute the combined loss
        mse_loss = mod3_mse_regularization(tf.squeeze(K.stack(regression_true)),
                                           tf.squeeze(K.stack(regression_pred)))
        # ========================================================================================================================

        # Compute the classification loss

        clf_v_loss = self.call_hierarchy_loss(tf.squeeze(classification_true[0]), tf.squeeze(classification_true[1]),
                                              tf.squeeze(classification_true[2]),
                                              tf.squeeze(classification_pred[0]), tf.squeeze(classification_pred[1]),
                                              tf.squeeze(classification_pred[2]))

        clf_d_loss = self.call_hierarchy_loss(tf.squeeze(classification_true[3]), tf.squeeze(classification_true[4]),
                                              tf.squeeze(classification_true[5]),
                                              tf.squeeze(classification_pred[3]), tf.squeeze(classification_pred[4]),
                                              tf.squeeze(classification_pred[5]))

        clf_j_loss = self.call_hierarchy_loss(None, tf.squeeze(classification_true[6]),
                                              tf.squeeze(classification_true[7]),
                                              None, tf.squeeze(classification_pred[6]),
                                              tf.squeeze(classification_pred[7]))

        classification_loss = self.v_class_weight * clf_v_loss + self.d_class_weight * clf_d_loss + self.j_class_weight * clf_j_loss

        # ========================================================================================================================
        # Compute the mutation oracle loss
        mutation_loss = K.binary_crossentropy(mutations_true,mutations_pred)

        # ========================================================================================================================



        # Combine the two losses using a weighted sum
        total_loss = ((self.regression_weight * mse_loss) +
                      (self.intersection_weight * total_intersection_loss)) + \
                     (self.classification_weight * classification_loss) +\
                     self.mutation_oracle_weight*mutation_loss

        return total_loss, total_intersection_loss, mse_loss, classification_loss,mutation_loss

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            loss, total_intersection_loss, mse_loss, classification_loss,mutation_oracle_loss\
                = self.multi_task_loss_v2(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.insec_loss_tracker.update_state(total_intersection_loss)
        self.mod3_mse_loss_tracker.update_state(mse_loss)
        self.total_ce_loss_tracker.update_state(classification_loss)
        self.mutation_oracle_loss_tracker.update_state(mutation_oracle_loss)
        # Return a dict mapping metric names to current value
        metrics = {m.name: m.result() for m in self.metrics}
        metrics['loss'] = self.loss_tracker.result()
        metrics['insec_loss'] = self.insec_loss_tracker.result()
        metrics['mod3_mse_loss'] = self.mod3_mse_loss_tracker.result()
        metrics['total_classification_loss'] = self.total_ce_loss_tracker.result()
        metrics['mutation_oracle'] = self.mutation_oracle_loss_tracker.result()
        return metrics

    def _freeze_segmentation_component(self):
        for layer in [
            self.concatenated_input_embedding,
            self.initial_embedding_attention,
            self.conv_layer_1,
            self.conv_layer_2,
            self.conv_layer_3,
            self.conv_layer_4,
            self.v_start_mid,
            self.v_start_out,
            self.v_end_mid,
            self.v_end_out,
            self.d_start_mid,
            self.d_start_out,
            self.d_end_mid,
            self.d_end_out,
            self.j_start_mid,
            self.j_start_out,
            self.j_end_mid,
            self.j_end_out,
        ]:
            layer.trainable = False

    def _freeze_v_classifier_component(self):
        for layer in [
            self.v_family_call_middle,
            self.v_family_call_head,
            self.v_gene_call_middle,
            self.v_gene_call_head,
            self.v_allele_call_middle,
            self.v_allele_feature_distill,
            self.v_allele_call_head
        ]:
            layer.trainable = False

    def _freeze_d_classifier_component(self):
        for layer in [
            self.d_family_call_middle,
            self.d_family_call_head,
            self.d_gene_call_middle,
            self.d_gene_call_head,
            self.d_allele_call_middle,
            self.d_allele_call_head
        ]:
            layer.trainable = False

    def _freeze_j_classifier_component(self):
        for layer in [
        self.j_gene_call_middle,
        self.j_gene_call_head,
        self.j_allele_call_middle,
        self.j_allele_call_head
        ]:
            layer.trainable = False

    def freeze_component(self, component):
        if component == ModelComponents.Segmentation:
            self._freeze_segmentation_component()
        elif component == ModelComponents.V_Classifier:
            self._freeze_v_classifier_component()
        elif component == ModelComponents.D_Classifier:
            self._freeze_d_classifier_component()
        elif component == ModelComponents.J_Classifier:
            self._freeze_j_classifier_component()

    def model_summary(self, input_shape):
        x = {'tokenized_sequence_for_masking': Input(shape=input_shape), 'tokenized_sequence': Input(shape=input_shape)}

        return Model(inputs=x, outputs=self.call(x)).summary()

    def plot_model(self, input_shape, show_shapes=True):
        x = {'tokenized_sequence_for_masking': Input(shape=input_shape), 'tokenized_sequence': Input(shape=input_shape)}
        return tf.keras.utils.plot_model(Model(inputs=x, outputs=self.call(x)), show_shapes=show_shapes)
