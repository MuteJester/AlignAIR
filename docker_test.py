import tensorflow as tf

from Trainer import Trainer
from VDeepJModel import VDeepJAllign
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau

# import time

# List available GPU devices
gpus = tf.config.experimental.list_physical_devices("GPU")

if not gpus:
    print("No GPUs available.")
else:
    for gpu in gpus:
        print(f"GPU: {gpu.name}")

# print("Hi")
