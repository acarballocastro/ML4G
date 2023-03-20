import pyBigWig
import numpy as np
import os
import pandas as pd
from utils import get_paths, get_gene_unique_name, merge_datasets, save_to_pickle, load_pickle
from tqdm import tqdm


# CAGE Preprocessing. We will add the gene expression level to the CAGE data.



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

    #Load the gex data for X3
    X3_gex = pd.read_table(f'{path}/X3_test_info.tsv')
    X3_gex.to_csv('../label_data/X3_test.tsv', sep="\t")


def prepare_train_validation(path):

    """
    Prepare the training data. Gives new names to the genes in each cell line and merges the data from both cell lines

    :param path: Path of the directory where the data is located

    :return:
    """
    cell_lines = ['X1', 'X2']
    file_type = ['train', 'val']
    get_gene_unique_name(path, cell_lines, file_type)

    files = ['train', 'val']

    for f in files:
        df1 = pd.read_table(path + '/X1_'+ f + '.tsv')
        df2 = pd.read_table(path + '/X2_' + f + '.tsv')
        X = pd.concat([df1, df2], ignore_index=True)
        X.to_csv('../label_data/X_' + f + '.tsv', sep="\t", index=False)


def prepare_test(path):

    """
    Prepare the test data. Gives new names to the genes in each cell line

    :param path: Path of the directory where the data is located

    :return:
    """
    cell_lines = ['X3']
    file_type = ['test']

    get_gene_unique_name(path, cell_lines, file_type)



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

    histones = ['H3K27ac','H3K27ac' ] #TODO: Add the rest of the histones and remove the duplicated one

    histone_data = np.zeros((0, n_bins))
    for histone in histones:

        path = f'../../histones/{histone}/X{cell_line}.bw' #TODO: custom this. Add the path where one has the histone data
        bw = pyBigWig.open(path) #Note:The chromosomes appear as chr1, chr2, etc. To access a range we need to use this: bw.stats("chr1",1,100, nBins=2)
        hist_stats = bw.stats(chr, start, end, nBins=n_bins)
        histone_data = np.vstack([histone_data, hist_stats])

    histone_data = histone_data.T

    return histone_data

def create_dataset(path: str, window_size: int, n_bins: int):
    """

    We will create the features dataset for the training, validation and testing.
    We will obtain a feature matrix (histone data) for each gene.
    We will use a window of 20000 bp before and after the TSS. #TODO: Decide number of bins
    We will store the gene name, its histone matrix and its gex (for X3 we don't have gex)

    :param path: Path of the directory where the data is located
    :param window_size: Number of bases we will take to the left and right of the TSS
    :param n_bins: Number of bins we will use to divide the window

    """

    files = ["X_train.tsv", "X_val.tsv", "X3_test.tsv"]

    for f in files:
        df = pd.read_table(path + f)
        if df.shape == len(df.gene_name.unique()): #There are no duplicated genes
            continue

        dataset = {}
        label ={}
        errors =0
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):

            start_pos = row.TSS_start - window_size
            end_pos = row.TSS_end + window_size
            cell_line = int(row.gene_name_unique.split('_')[1].split('X')[1])

            try:
                histone_data = preprocess_histone_data(cell_line, row["chr"], start_pos, end_pos, n_bins)
            except RuntimeError:
                errors +=1
                continue

            if cell_line != 3:
                dataset[row["gene_name_unique"]] = [histone_data, row["gex_transf"]]
            else:
                dataset[row["gene_name_unique"]] = [histone_data]

        save_to_pickle(path, f, dataset)


if __name__ == '__main__':

    path_data= "../label_data"
    path_expression = "../CAGE-train"
    if not os.path.exists(path_data):
        os.makedirs(path_data)
        preprocess_cage_data(path_expression)

        # Prepare the data for training and validation
        prepare_train_validation(path_data)

        # Prepare the data for testing
        prepare_test(path_data)

    create_dataset("../label_data/", 20000, 100)




