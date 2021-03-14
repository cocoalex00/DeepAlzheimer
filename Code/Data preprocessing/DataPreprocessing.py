#!usr/bin/python3

#This code will preprocess both datasets used (DNA methylation and Gene expression) in order to achieve the final multi-omics dataset that will be used throughout the project.

import pandas as pd
from pathlib import Path

#loading both datasets into dataframes 
GeneExpression = pd.read_table('allforDNN_ge_sample.tsv')
DNAmethylation = pd.read_table('allforDNN_me_sample.tsv')


#Separating AD positive and negative samples in both datasets for merging

GEpos = GeneExpression.where(GeneExpression["Label_AD"] == 1).dropna(0,'all')
GEneg = GeneExpression.where(GeneExpression["Label_AD"] == 0).dropna(0,'all')

DNApos = DNAmethylation.where(DNAmethylation["Label_AD"] == 1).dropna(0,'all')
DNAneg = DNAmethylation.where(DNAmethylation["Label_AD"] == 0).dropna(0,'all')

print("AD positive and negative samples extracted from both datasets: ")
print("Gene expression positive dataset's shape: " + str(GEpos.shape))
print("Gene expression negative dataset's shape: " + str(GEneg.shape))
print("DNA methylation positive dataset's shape: " + str(DNApos.shape))
print("DNA methylation negative dataset's shape: " + str(DNAneg.shape))



#Creating all the combinations of AD positive and negative samples
#The columns of sampleID need to be deleted from both datasets but the labels only need to be erased from one as they are needed in the final table
ADpos = pd.merge(GEpos.drop(['SampleID','Label_AD','Label_No'],1),DNApos.drop(['SampleID'],1), how='cross')

ADneg = pd.merge(GEneg.drop(['SampleID','Label_AD','Label_No'],1),DNAneg.drop(['SampleID'],1), how='cross')

print('-'*10) 
print("All combinations achieved: ")
print("AD negative dataset's shape: " + str(ADneg.shape))
print("AD positive dataset's shape: " + str(ADpos.shape))


#Merging AD possitive and negative samples to form the final dataset

MultiOmicsDataset = pd.concat([ADpos, ADneg])

print('-'*10) 
print("Multi-Omics dataset created: ")
print("Multi-Omics dataset's shape: " + str(MultiOmicsDataset.shape))
print(MultiOmicsDataset)


#Exporting the multi-omics dataset to a csv file 

print('-'*10)
print("Converting the dataframe to a .csv file, please wait")
MultiOmicsDataset.to_csv(r'Output/MultiOmicsDataset.csv')

print("Conversion complete")
