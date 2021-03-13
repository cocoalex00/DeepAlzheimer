#!usr/bin/python3

#This is a raw first implementation of the feature selection method using a binary coded genetic algorithm, the parallelised version will be based on this.

import pandas as pd
from pathlib import Path

#loading both datasets into dataframes 
GeneExpression = pd.read_table('allforDNN_ge_sample.tsv')
DNAmethylation = pd.read_table('allforDNN_me_sample.tsv')


#Separating AD positive and negative samples for merging

GEpos = GeneExpression.where(GeneExpression["Label_AD"] == 1).dropna(0,'all')
GEneg = GeneExpression.where(GeneExpression["Label_AD"] == 0).dropna(0,'all')

DNApos = DNAmethylation.where(DNAmethylation["Label_AD"] == 1).dropna(0,'all')
DNAneg = DNAmethylation.where(DNAmethylation["Label_AD"] == 0).dropna(0,'all')

print(GEpos)
print(GEneg)
print(GeneExpression)
