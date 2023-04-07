import os
import pandas as pd
import numpy as np
import pickle

def get_paths(path):
    """
    The function will return the whole path for all the files in the specified directory.

    :param path: This is the path of the directory
    """
    files = os.listdir(path)
    return [os.path.join(path, f) for f in files]

def merge_datasets(df1, df2):

    """
    Merge two datasets using the gene name

    """
    df_merge = df1.set_index('gene_name').join(df2.set_index('gene_name'))

    return df_merge

def check_genes_in_cl():

    """
    The function will check if there are matching genes with different expression levels in the X1 and X2 cell lines

    We find out that genes are the same in both cell lines, but the expression levels are different.
    """
    X1_train = pd.read_table('../label_data/X1_train.tsv')
    X2_train = pd.read_table('../label_data/X2_train.tsv')
    X1_val = pd.read_table('../label_data/X1_val.tsv')
    X2_val = pd.read_table('../label_data/X2_val.tsv')

    X1_train_genes = X1_train['gene_name'].values
    X2_train_genes = X2_train['gene_name'].values
    X1_val_genes = X1_val['gene_name'].values
    X2_val_genes = X2_val['gene_name'].values

    X1_genes = np.concatenate((X1_train_genes, X1_val_genes))
    X2_genes = np.concatenate((X2_train_genes, X2_val_genes))

    X1_genes = np.unique(X1_genes)
    X2_genes = np.unique(X2_genes)

    X1_genes = set(X1_genes)
    X2_genes = set(X2_genes)

    print("Number of genes in X1: ", len(X1_genes))
    print("Number of genes in X2: ", len(X2_genes))

    print("Number of genes in X1 and X2: ", len(X1_genes.intersection(X2_genes)))

    print("Number of genes in X1 but not in X2: ", len(X1_genes.difference(X2_genes)))

    print("Number of genes in X2 but not in X1: ", len(X2_genes.difference(X1_genes)))

def get_gene_unique_name(path, cell_lines: list, file_type: list):

    """
    This function will rename the genes as: (gene_name)_(cell_line). We need to create unique gene names for each cell
    line as the genes are the same but with different expression levels. We are just interested in studying how the
    histone marks relate to the gene expression so we don't really care about the exact name of the gene

    :param path: Path of the CAGE directory
    :param cell_line: List of cell lines
    :param file_type: List of file types
    :return:
    """

    for cell_line in cell_lines:
        for type_f in file_type:

            var_name= f'{cell_line}_{type_f}'
            df = pd.read_table(f'{path}/{var_name}.tsv')
            df['gene_name_unique'] = df['gene_name'].apply(lambda x: f'{x}_{cell_line}')

            df.to_csv(f'{path}/{var_name}.tsv', sep='\t', index=False)


def save_to_pickle(path, f, dict):
    """
    Saves a dictionary to pickle format for later use with pytorch

    """

    with open(path + f.split('.')[0] + '.pickle', 'wb') as f:
        pickle.dump(dict, f)

def load_pickle(path, f):

    """
    Loads a pickle file
    :param path:
    :return: dictionary stored in the pickle file
    """

    with open(path + '/' + f , 'rb') as f:

        # load the object from the file
        dict = pickle.load(f)

    return dict