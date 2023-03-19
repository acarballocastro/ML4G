import pyBigWig
import numpy as np
import os
import pandas as pd
from utils import get_paths

# CAGE Preprocessing. We will add the gene expression level to the CAGE data.

def merge_datasets(df1, df2):

    """
    Merge two datasets using the gene name

    """
    df_merge = df1.set_index('gene_name').join(df2.set_index('gene_name'))

    return df_merge

def preprocess_cage_data(path):

    """
    Preprocess CAGE data by adding the gex column to the information file

    :param path: Path of the CAGE directory
    """

    file_paths = get_paths(path)
    var_names = [f.split('/')[-1].split('.')[0] for f in file_paths]
    file_dict = {}

    # Create a dictionary with the file names as keys and the file paths as values
    for f_path, f_name in zip(file_paths, var_names):
        file_df = pd.read_table(f_path)
        file_dict[f_name] = file_df

    # Merge data
    X1_train = merge_datasets(file_dict['X1_train_info'], file_dict['X1_train_y'])
    X1_val = merge_datasets(file_dict['X1_val_info'], file_dict['X1_val_y'])
    X2_train = merge_datasets(file_dict['X2_train_info'], file_dict['X2_train_y'])
    X2_val = merge_datasets(file_dict['X2_val_info'], file_dict['X2_val_y'])

    # log2 transform with pseudocount and save data

    names= ["X1_train","X1_val", "X2_train", "X2_val"]
    for idx, name in enumerate([X1_train, X1_val, X2_train, X2_val]):
        name["gex_transf"] = np.log2(name["gex"] + 1)
        name.to_csv( '../label_data/' + names[idx] + '.tsv', sep="\t")

#TODO: We need to locate for each gene the position of each histone. We need to do a function for this
def preprocess_histone_data(cell_line: int, chr: str, start: int, end:int, n_bins:int):

    """
    Process the data from the different histones and creates a matrix of n_bins x n_histones dimensions
    This function will be used for each gene in the dataset. The histone data will be taken from the range on the TSS of the gene

    :param cell_line: The cell line we are studying. It has to be 1,2 or 3
    :param chr: Chromosome of where the sample is located
    :param start: TSS for the gene
    :param end: TSS for the gene
    : n_bins: Number of bins. We will get the average from each bin
    :return: Histone data matrix (n_bins x n_histones). Each entry is the average value of the bigWig measurement in the bin.
    """

    histones = ['H3K27ac','H3K27ac' ] #TODO: Add the rest of the histones

    histone_data = np.zeros((0, n_bins))
    for histone in histones:

        path = f'../../histones/{histone}/X{cell_line}.bw' #TODO: custom this
        bw = pyBigWig.open(path)#TODO:The chromosomes appear as chr1, chr2, etc. To access a range we need to use this: bw.stats("chr1",1,100, nBins=2)
        hist_stats = bw.stats(chr, start, end, nBins=n_bins)
        histone_data = np.vstack([histone_data, hist_stats])

    histone_data = histone_data.T

    return histone_data

def create_dataset():
    """

    We will create the dataset for the training and the validation. We will obtain a feature matrix (histone data) for each gene.
    We will use a window of 40000 bp before and after the TSS. #TODO: Decide number of bins
    We will store the gene name, its histone matrix and its gex (for X3 we don't have gex)

    :return:
    """

if __name__ == '__main__':
    if not os.path.exists('../label_data'):
        os.makedirs('../label_data')
        preprocess_cage_data("../CAGE-train")

    #histone_data = preprocess_histone_data(1, "chr1", 1, 100,10)


