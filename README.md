# Machine Learning for Genomics - Project 1

How to run this code:

## Enviroment and requirements

The files `environment.yml` and `requirements.txt` contain the information of packages necessary to run this code.

## Data preprocesing

The script `dataset.py` in the folder *preprocessing* includes the steps needed to prepare the data that will then be used for model training and testing.

- Data for gene expression should be in the folder *CAGE-train*
- Data of the histones should be in the folder *histones*. In this folder, there should exist a subdirectory for each of the histones to be used with the name of the corresponding histone.
- Resulting datasets wil be stored in the folder *data*

Command for runing:

    ```
    python dataset.py
    ```

Histones used are: H3K4me1, H3K4me3, H3K9me3, H3K27ac, H3K27me3

## Model training and evaluation

The script `model.py` in the folder *model* includes the code to train the model and test it. Data is read from the folder *data*. Predictions will be stored in the folder ...

Command for runing:

    ```
    python model.py
    ```

The script `model_wandb.py` in the folder *model* is the version used for hyperparmeter tuning using the library `wandb`.
