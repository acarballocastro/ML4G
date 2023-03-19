import pandas as pd
import numpy as np

# CAGE Preprocessing 

## Read in data
X1_train_info = pd.read_table("../CAGE-train/X1_train_info.tsv")
X1_train_y = pd.read_table("../CAGE-train/X1_train_y.tsv")
X1_val_info = pd.read_table("../CAGE-train/X1_val_info.tsv")
X1_val_y = pd.read_table("../CAGE-train/X1_val_y.tsv")
X2_train_info = pd.read_table("../CAGE-train/X2_train_info.tsv")
X2_train_y = pd.read_table("../CAGE-train/X1_train_y.tsv")
X2_val_info = pd.read_table("../CAGE-train/X2_val_info.tsv")
X2_val_y = pd.read_table("../CAGE-train/X1_val_y.tsv")
X3_test_info = pd.read_table("../CAGE-train/X3_test_info.tsv")

## Merge data

X1_train = X1_train_info.set_index('gene_name').join(X1_train_y.set_index('gene_name'))
X1_val = X1_val_info.set_index('gene_name').join(X1_val_y.set_index('gene_name'))
X2_train = X2_train_info.set_index('gene_name').join(X2_train_y.set_index('gene_name'))
X2_val = X2_val_info.set_index('gene_name').join(X2_val_y.set_index('gene_name'))

## log2 transform with pseudocount

X1_train["gex_transf"] = np.log2(X1_train["gex"] + 1)
X1_val["gex_transf"] = np.log2(X1_val["gex"] + 1)
X2_train["gex_transf"] = np.log2(X2_train["gex"] + 1)
X2_val["gex_transf"] = np.log2(X2_val["gex"] + 1)

## Save data

X1_train.to_csv('../CAGE-train/X1_train.tsv', sep="\t")
X1_val.to_csv('../CAGE-train/X1_val.tsv', sep="\t")
X2_train.to_csv('../CAGE-train/X2_train.tsv', sep="\t")
X2_val.to_csv('../CAGE-train/X2_val.tsv', sep="\t")