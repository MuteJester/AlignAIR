# AlignAIRR

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Introduction

This repository provides a Python implementation of the VDeepJ architecture model trainer, which allows you to train and predict using the VDeepJ model.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Parameters](#parameters)
  - [Creating a Trainer Instance and Training](#creating-a-trainer-instance-and-training)
  - [Predicting using the Trainer class](#predicting-using-the-trainer-class)
  - [Saving and Loading a Model](#saving-and-loading-a-model)
  - [Plotting the Training History](#plotting-the-training-history)
- [License](#license)

## Prerequisites

Before using the VDeepJ model trainer, make sure you have the following prerequisites installed:

- Python 3.8
- numpy==1.21.5
- tensorflow==2.5.0
- tensorflow-addons==0.16.1
- tensorflow-gpu==2.9.1
- pandas==1.3.5
- scipy==1.9.1
- seaborn==0.12.1
- see requirements.txt for more detailed list


## Usage

## Parameters

TODO: @raneis --> Consider adding examples as some of the parameters are hard to understand (what is the **model** pararameter?)

- **model**: The model parameter represents the one of the VDeepJ model architectures.
- **epochs**: The epochs parameter specifies the number of training iterations.
- **batch_size**: The batch_size parameter determines the number of samples processed at once.
- **train_dataset**: The train_dataset parameter represents the path to the dataset used for training.
- **input_size**: The input_size parameter defines the length of the input sequences (default 512).
- **log_to_file**: A boolean flag to log the training progress to a file.
- **log_file_name**: The name of the log file where the training progress will be recorded.
- **log_file_path**: The directory path where the log file will be stored.
- **corrupt_beginning**: A boolean flag to artificially corrupt the beginning of input sequences during training.
- **classification_head_metric**: The evaluation metric used for the classification head of the VDeepJ model.
- **interval_head_metric**: The evaluation metric used for the interval head of the VDeepJ model.
- **corrupt_proba**: The probability of corrupting each input sequence during training.
- **verbose**: The verbosity level of the training process.
- **nucleotide_add_coef**: The coefficient used to control the addition of nucleotides during corruption.
- **nucleotide_remove_coef**: The coefficient used for removing nucleotides during corruption.
- **chunked_read**: A boolean flag indicating whether the input sequences are read and processed in smaller chunks.
- **pretrained**: The path to pre-trained weights of the VDeepJ model.
- **compute_metrics**: Additional metrics to be computed during training.
- **callbacks**: TensorFlow Keras callbacks applied during training.
- **optimizers**: The optimizer used for training the model.
- **optimizers_params**: Additional parameters for the optimizer.

### Creating a Trainer Instance and Training a model

To create a Trainer instance and train the VDeepJ model, follow these steps:

```python
# Import the necessary libraries
import tensorflow as tf
from Trainer import Trainer
from VDeepJModel import VDeepJAllign


# Define your model
model = VDeepJAllign()
# Define other parameters
epochs = 5
batch_size = 64
train_dataset = r'E:\my_path\train_data.tsv'
# Create a Trainer instance with desired parameters
trainer = Trainer(
 model=model,
 epochs=epochs,
 batch_size=batch_size,
 train_dataset=train_dataset,
 ...
)

# Train the model
trainer.train()

```

## Predicting using the Trainer class


To make predictions using the trained `Trainer` object, you can use the following code snippet:

```python
predicted_pp = trainer.predict(r'E:\my_path\target_data.tsv', raw=False, top_k=3)
```
The predict method accepts the following parameters:

TODO: @raneisn --> Consider adding a csv input example, ephasize that we are using Ayelets V groups

- **r'E:\my_path\target_data.tsv'**: The path to the test set file that contains the input sequences for prediction.
- **raw**: A boolean parameter that determines the format of the prediction output. If raw is set to True, the method will return unprocessed model output. If raw is set to False, the output will be converted into a DataFrame, which is easier to work with and understand.
- **top_k**: An integer parameter that controls the number of results returned for each classification head.

#### Example Output:

| v_family |        v_family_scores |                                            v_gene | v_gene_scores |                                          v_allele | v_allele_scores |                                          d_family |     d_family_scores |                                            d_gene | d_gene_scores |                                               ... | j_gene |       j_gene_scores |                                          j_allele | j_allele_scores |                                             j_end | d_start | d_end | v_start | v_end | j_start |     |
|---------:|-----------------------:|--------------------------------------------------:|--------------:|--------------------------------------------------:|----------------:|--------------------------------------------------:|--------------------:|--------------------------------------------------:|--------------:|--------------------------------------------------:|-------:|--------------------:|--------------------------------------------------:|----------------:|--------------------------------------------------:|--------:|------:|--------:|------:|--------:|-----|
|        0 | IGHVF2\|IGHVF8\|IGHVF1 | [0.1755390167236328, 0.14425861835479736, 0.14... |   G35\|G6\|G5 | [0.41292786598205566, 0.28214359283447266, 0.1... |      01\|02\|03 | [0.574677586555481, 0.29233354330062866, 0.058... | IGHD3\|IGHD2\|IGHD1 | [0.177727609872818, 0.16977187991142273, 0.166... |     9\|23\|26 | [0.06180494278669357, 0.06030962988734245, 0.0... |    ... | IGHJ5\|IGHJ4\|IGHJ1 | [0.2393306940793991, 0.21580205857753754, 0.17... |      02\|01\|03 | [0.6055132150650024, 0.2594718337059021, 0.113... |     337 |   287 |     295 |     0 |     283 | 300 |
|        1 | IGHVF2\|IGHVF8\|IGHVF5 | [0.19508720934391022, 0.1810883730649948, 0.14... |   G25\|G2\|G4 | [0.18179909884929657, 0.08164383471012115, 0.0... |      01\|03\|02 | [0.24517884850502014, 0.1757553368806839, 0.13... | IGHD3\|IGHD2\|IGHD6 | [0.17414222657680511, 0.17128515243530273, 0.1... |    27\|23\|26 | [0.06811603903770447, 0.0616660974919796, 0.06... |    ... | IGHJ5\|IGHJ1\|IGHJ4 | [0.2613930106163025, 0.23604996502399445, 0.19... |      02\|01\|03 | [0.6257255673408508, 0.2849787771701813, 0.065... |     333 |   283 |     292 |     0 |     280 | 296 |
|        2 | IGHVF2\|IGHVF5\|IGHVF8 | [0.18603621423244476, 0.17141517996788025, 0.1... | G25\|G10\|G13 | [0.16186371445655823, 0.060858651995658875, 0.... |      03\|01\|02 | [0.2615293860435486, 0.2521117925643921, 0.207... | IGHD3\|IGHD2\|IGHD6 | [0.17427770793437958, 0.1709291636943817, 0.16... |     27\|9\|23 | [0.06502501666545868, 0.06245790049433708, 0.0... |    ... | IGHJ5\|IGHJ4\|IGHJ1 | [0.2446955293416977, 0.19809859991073608, 0.18... |      02\|01\|03 | [0.6446681022644043, 0.2810792624950409, 0.063... |     330 |   280 |     289 |     0 |     278 | 293 |

## Saving and Loading a Model
### To save the trained Trainer object, you can use the following methods:
#### Save Model Weights
```python
trainer.save_model('/savepath')
```
This function saves the trained model weights and configuration to the specified directory path.
(it will be saved as a weight file with  a unique uuid), the function return the file name after it is saved

#### Save Dataset Object

```python
trainer.save_dataset_object('path/filename.ds')
```
This function saves the dataset object used by the Trainer to the specified file path.
The dataset object contains all important data preprocessing parameters derived for that model.


### To load the saved Trainer object and continue training or make predictions, follow these steps:

1. Create a new Trainer instance:
```python

lt = Trainer(VDeepJAllign(), epochs=1, batch_size=64, verbose=1,
             log_to_file=True, log_file_name='log', log_file_path='log_save_file_path')
```
2. Load the dataset object:
```python
lt.load_dataset_object('path/filename.ds')
```
3. Rebuild the model:
```python
lt.rebuild_model()
```
This rebuilds the model architecture based on the loaded dataset object.

4. Load the saved model weights:

```python
lt.load_model('savepath/weight_file_name')
```
## Plotting the Training History
```python
trainer.plot_history()
```
You can add a save path as a parameter to save the figure,
in general after each call to `.train()` a history object is created/
updated as an attribute of the `Trainer` object, use
`trainer.history` to access it.
