class ExtractGeneMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ExtractGeneMask, self).__init__(**kwargs)
        self.mul_t = Multiply()  # ([input_t,mid_cu])
        self.mul_c = Multiply()  # ([input_c,mid_cu])
        self.mul_g = Multiply()  # ([input_g,mid_cu])
        self.mul_a = Multiply()  # ([input_a,mid_cu])

    def call(self, inputs):
        (a, t, g, c), mask = inputs

        masked_a = self.mul_a([a, mask])
        masked_t = self.mul_t([t, mask])
        masked_g = self.mul_g([g, mask])
        masked_c = self.mul_c([c, mask])

        return masked_a, masked_t, masked_g, masked_c


class ExtractGeneMask1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ExtractGeneMask1D, self).__init__(**kwargs)
        self.mul_seq = Multiply()  # ([input_t,mid_cu])

    def call(self, inputs):
        seq, mask = inputs

        masked_seq = self.mul_seq([seq, mask])

        return masked_seq


class ExtractedMaskEncode(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ExtractedMaskEncode, self).__init__(**kwargs)
        self.conv_1 = Conv1D(16, 3, padding='same', activation='relu', kernel_initializer='he_uniform')
        self.maxpool_1 = MaxPool1D(2)
        self.conv_2 = Conv1D(32, 3, padding='same', activation='relu', kernel_initializer='he_uniform')
        self.attention_1 = Attention()
        self.conv_3 = Conv1D(8, 8, padding='same', activation='relu', kernel_initializer='he_uniform')
        self.maxpool_2 = MaxPool1D(2)
        self.conv_4 = Conv1D(32, 3, padding='same', activation='relu', kernel_initializer='he_uniform')
        self.attention_2 = Attention()
        self.conv_5 = Conv1D(8, 8, padding='same', activation='relu', kernel_initializer='he_uniform')
        self.maxpool_3 = MaxPool1D(2)

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.maxpool_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.attention_1([x, x])
        x = self.maxpool_2(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.attention_2([x, x])
        x = self.maxpool_3(x)
        return x


class EncodingConcatBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EncodingConcatBlock, self).__init__(**kwargs)
        self.concated = concatenate
        self.maxpool_1 = MaxPool1D(3)
        self.attention = Attention()
        self.flatten = Flatten()
        self.dense_1 = Dense(512, activation='relu')
        self.dropout_1 = Dropout(0.5)

    def call(self, inputs):
        x = self.concated(inputs)
        x = self.maxpool_1(x)
        x = self.attention([x, x])
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dropout_1(x)
        return x


class Conv2D_and_BatchNorm(tf.keras.layers.Layer):
    def __init__(self, filters=16, kernel=(3, 3), max_pool=(2, 1), **kwargs):
        super(Conv2D_and_BatchNorm, self).__init__(**kwargs)
        self.conv_2d = Conv2D(filters, kernel, padding='same', kernel_initializer='he_uniform')  # (inputs)
        self.batch_norm = BatchNormalization()  # (concatenated_path)
        self.activation = LeakyReLU()  # (concatenated_path)
        self.max_pool = MaxPool2D(max_pool)  # (concatenated_path)

    def call(self, inputs):
        x = self.conv_2d(inputs)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x


class Conv1D_and_BatchNorm(tf.keras.layers.Layer):
    def __init__(self, filters=16, kernel=3, max_pool=2, **kwargs):
        super(Conv1D_and_BatchNorm, self).__init__(**kwargs)
        self.conv_2d = Conv1D(filters, kernel, padding='same', kernel_initializer='he_uniform')  # (inputs)
        self.batch_norm = BatchNormalization()  # (concatenated_path)
        self.activation = LeakyReLU()  # (concatenated_path)
        self.max_pool = MaxPool1D(max_pool)  # (concatenated_path)

    def call(self, inputs):
        x = self.conv_2d(inputs)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        return x


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, emded_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=emded_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=emded_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def mod3_mse_loss(y_true, y_pred):
    y_true_casted = K.cast(y_true, dtype='float32')
    # Compute the residual of the predicted values modulo 3

    div_pred = K.cast(y_pred // 3.0, dtype='float32')
    mod_pred = y_pred - div_pred * 3

    div_true = K.cast(y_true_casted // 3.0, dtype='float32')
    mod_true = y_true_casted - div_true * 3

    residual = K.abs(mod_pred - mod_true)

    # Penalize the deviation from the nearest multiple of 3
    mod3_loss = K.mean(K.switch(residual <= 1, residual, 3 - residual))

    # Compute the MSE loss
    mse_loss = K.mean(K.square(y_pred - y_true_casted))

    # Combine the two losses using a weighted sum
    mod3_weight, mse_weight = 0.3, 0.7  # adjust the weights as needed
    total_loss = mod3_weight * mod3_loss + mse_weight * mse_loss

    return total_loss


def mod3_mse_regularization(y_true, y_pred):
    """
    Computes the mean squared error loss with regularization to enforce predictions to be modulo 3.
    """
    mse = K.mean(K.square(y_true - y_pred))
    mod_pred = K.abs(y_pred - 3 * K.cast(K.round(y_pred / 3), 'float32'))
    mod_loss = K.mean(K.square(mod_pred))
    return mse + mod_loss
