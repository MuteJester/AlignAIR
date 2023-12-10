import tensorflow.keras.backend as K


def swish(x):
    return K.sigmoid(x) * x


def mish(x):
    return x * K.tanh(K.softplus(x))

