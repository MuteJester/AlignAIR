import tensorflow as tf
import itertools
import numpy as np

class CharNGramVectorizer(tf.keras.layers.Layer):
    """
    A TensorFlow Layer that applies a character n-gram 'CountVectorizer' logic
    with binary=True. For each input string in the batch, it generates all
    character n-grams of length [min_ngram..max_ngram], looks them up in a fixed
    vocabulary, and returns a dense [batch_size, vocab_size] float Tensor with
    1.0 if that n-gram appears at least once, otherwise 0.0.
    """

    def __init__(self, min_ngram=3, max_ngram=5, binary=True, **kwargs):
        super().__init__(**kwargs)
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.binary = binary

        # 1) Build a vocabulary of all possible n-grams in [min_ngram..max_ngram].
        self.vocab_plain = [
            ''.join(p)
            for n in range(self.min_ngram, self.max_ngram + 1)
            for p in itertools.product('ATCGN', repeat=n)
        ]
        self.vocab_size = len(self.vocab_plain)

        # 2) Create a lookup table: n-gram string -> index in [0..vocab_size-1].
        vocab_tensor = tf.constant(self.vocab_plain, dtype=tf.string)
        indices = tf.range(self.vocab_size, dtype=tf.int64)
        kv_init = tf.lookup.KeyValueTensorInitializer(vocab_tensor, indices)
        self.lookup_table = tf.lookup.StaticHashTable(
            initializer=kv_init,
            default_value=-1  # -1 indicates "not found" / OOV
        )

    def call(self, inputs):
        """
        Args:
          inputs: A string Tensor of shape [batch_size], each entry is one DNA sequence.
        Returns:
          A float Tensor of shape [batch_size, vocab_size], with 1.0 if that n-gram
          appears at least once for the sample, else 0.0.
        """
        if isinstance(inputs, (list, tuple)):
            inputs = tf.convert_to_tensor(inputs, dtype=tf.string)

        # 1) Split each string into characters => Ragged[batch_size, (num_chars)].
        chars_ragged = tf.strings.unicode_split(inputs, 'UTF-8')

        # 2) For each n in [min_ngram..max_ngram], build n-grams and concatenate rows.
        ngram_list = []
        for n in range(self.min_ngram, self.max_ngram + 1):
            ngrams_ragged = tf.strings.ngrams(chars_ragged, n, separator='')
            ngram_list.append(ngrams_ragged)
        # merged: Ragged[batch_size, sum_of_ngrams_per_row]
        combined_ngrams = tf.concat(ngram_list, axis=1)

        # 3) Flatten all n-grams to 1D => shape [total_ngrams_in_batch]
        flat_ngrams = combined_ngrams.flat_values

        # 4) Lookup each n-gram in the vocabulary. -1 => not found/OOV
        vocab_indices = self.lookup_table.lookup(flat_ngrams)

        # 5) Convert row_splits => segment_ids so we know which sample each n-gram belongs to
        row_splits = combined_ngrams.row_splits  # shape [batch_size+1]
        segment_ids = tf.ragged.row_splits_to_segment_ids(row_splits)  # shape [total_ngrams_in_batch]

        # 6) Filter out OOV n-grams
        valid_mask = (vocab_indices >= 0)
        valid_rows = tf.boolean_mask(segment_ids, valid_mask)
        valid_cols = tf.boolean_mask(vocab_indices, valid_mask)

        # ---------------------------
        # 7) Remove duplicates in (row, col) to ensure each (row, col) is only used once.
        #    For a purely binary presence, we only need 1 occurrence of each (row, col).
        #    We'll encode (row, col) into a single integer, then call tf.unique.
        #    row * vocab_size + col => unique integer. Then decode back to row, col.
        # ---------------------------
        rowcol_packed = valid_rows * self.vocab_size + valid_cols
        unique_rowcol = tf.unique(rowcol_packed).y  # shape [?]
        # decode:
        unique_rows = unique_rowcol // self.vocab_size
        unique_cols = unique_rowcol % self.vocab_size

        # 8) Build a SparseTensor => shape [batch_size, vocab_size], with 1 at (row, col)
        coords = tf.stack([unique_rows, unique_cols], axis=1)
        values = tf.ones_like(unique_rows, dtype=tf.float32)
        batch_size = tf.shape(inputs)[0]

        sparse_tensor = tf.sparse.SparseTensor(
            indices=coords,
            values=values,
            dense_shape=[batch_size, self.vocab_size]
        )

        # 9) Reorder & convert to dense => final [batch_size, vocab_size]
        sparse_tensor = tf.sparse.reorder(sparse_tensor)
        result = tf.sparse.to_dense(sparse_tensor, default_value=0.0)
        return result

    def get_config(self):
        config = super().get_config()
        config.update({
            "min_ngram": self.min_ngram,
            "max_ngram": self.max_ngram,
            "binary": self.binary,
            "vocab_size": self.vocab_size,
        })
        return config

    def compute_output_shape(self, input_shape):
        # input_shape is [batch_size] of strings => output is [batch_size, vocab_size]
        return (input_shape[0], self.vocab_size)

class LogisticRegressionLayer(tf.keras.layers.Layer):
    """
    A custom Keras layer implementing multinomial Logistic Regression.
    It takes an input of shape [batch_size, input_dim] and produces
    a probability distribution over `num_classes`.

    Args:
        label_map: A dict mapping integer class IDs -> string label names
                   (e.g. {0: 'Normal', 1: 'Complement', ...}).
    """
    def __init__(self, label_map,vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.label_map = label_map
        self.vocab_size=vocab_size
        self.num_classes = len(label_map)

    def build(self, input_shape):
        """
        Called once automatically to create weights, after input shape is known.
        """
        # input_shape is typically (batch_size, input_dim)
        input_dim = self.vocab_size

        # Weights: shape (input_dim, num_classes)
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.num_classes),
            initializer="glorot_uniform",
            trainable=True
        )
        # Bias: shape (num_classes,)
        self.bias = self.add_weight(
            name="bias",
            shape=(self.num_classes,),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        """
        Forward pass:  logits = inputs * W + b
                       y_prob = softmax(logits)
        """
        logits = tf.matmul(inputs, self.kernel) + self.bias
        return tf.nn.softmax(logits)

    def get_config(self):
        """
        Needed if you want to serialize this layer or save it in a model.
        """
        config = super().get_config()
        config.update({
            "label_map": self.label_map
        })
        return config
