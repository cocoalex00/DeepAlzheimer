#!usr/bin/python3

#This code will preprocess both datasets used (DNA methylation and Gene expression) in order to achieve the final multi-omics datasets that will be used throughout the project.

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

# Separating the positive and negative samples for training/testing before creating all posible combinations. 70-30%

GEposTr = GEpos.iloc[:307,:] # Tr = training
GEposTe = GEpos.iloc[307:,:] # Te = testing

GEnegTr = GEneg.iloc[:192,:]
GEnegTe = GEneg.iloc[192:,:]

DNAposTr = DNApos.iloc[:51,:]
DNAposTe = DNApos.iloc[51:,:]

DNAnegTr = DNAneg.iloc[:47,:]
DNAnegTe = DNAneg.iloc[47:,:]


#Creating all the combinations of AD positive and negative samples
#The columns of sampleID need to be deleted from both datasets but the labels only need to be erased from one as they are needed in the final table

ADposTr = pd.merge(GEposTr.drop(['SampleID','Label_AD','Label_No'],1),DNAposTr.drop(['SampleID'],1), how='cross')
ADnegTr = pd.merge(GEnegTr.drop(['SampleID','Label_AD','Label_No'],1),DNAnegTr.drop(['SampleID'],1), how='cross')

ADposTe = pd.merge(GEposTe.drop(['SampleID','Label_AD','Label_No'],1),DNAposTe.drop(['SampleID'],1), how='cross')
ADnegTe = pd.merge(GEnegTe.drop(['SampleID','Label_AD','Label_No'],1),DNAnegTe.drop(['SampleID'],1), how='cross')

 
print('-'*100) 
print("All combinations achieved: ")
print("AD negative dataset's shape(Training): " + str(ADnegTr.shape))
print("AD negative dataset's shape(Testing): " + str(ADnegTe.shape))
print("AD positive dataset's shape(Training): " + str(ADposTr.shape))
print("AD positive dataset's shape(Testing): " + str(ADposTe.shape))


#Merging AD possitive and negative samples to form the final datasets

MultiOmicsTrain = pd.concat([ADposTr, ADnegTr])
MultiOmicsTest = pd.concat([ADposTe, ADnegTe])

print('-'*100) 
print("Multi-Omics dataset created: ")
print("Multi-Omics dataset's shape (Training): " + str(MultiOmicsTrain.shape))
print(MultiOmicsTrain)
print("Multi-Omics dataset's shape (Testing): " + str(MultiOmicsTest.shape))
print(MultiOmicsTest)

# Shuffling the datasets 

MultiOmicsTrain = MultiOmicsTrain.sample(frac = 1)
MultiOmicsTest = MultiOmicsTest.sample(frac = 1)
print('-'*100) 
print("Multi-Omics dataset shuffled: ")
print("Multi-Omics dataset(train):")
print(MultiOmicsTrain)
print("Multi-Omics dataset(test):")
print(MultiOmicsTest)
# Exporting the multi-omics datasets to csv files

print('-'*100)
print("Converting the dataframes to.csv files, please wait")
MultiOmicsTrain.to_csv(r'Output/MultiOmicsTrain.csv')
MultiOmicsTest.to_csv(r'Output/MultiOmicsTest.csv')
print("Conversion complete")
