#%%
import pandas as pd
import numpy as np
from collections import Counter

# 1. drop unexpressed genes
# constants
# file_path = f"/isilon/datalake/cialab/original/cialab/image_database/d00154/Tumor_gene_counts/All_countings/training_data_17_tumors_31_classes.csv"
# reads_cutoff=100
# minimum_expressed_samples=40

def find_expressed_genes(file_path, reads_cutoff=100, minimum_expressed_samples=40):
    data_dir = file_path
    df = pd.read_csv(data_dir, header=[0, 1], index_col=0)
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    expressed_genes = (df >= reads_cutoff).sum(axis=1) >= minimum_expressed_samples
    return expressed_genes


# %%
