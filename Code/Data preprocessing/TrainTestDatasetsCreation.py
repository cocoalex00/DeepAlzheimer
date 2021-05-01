#!usr/bin/python3

#This code will separate the Multi-Omics Dataset in train and test samples/labels.

import pandas as pd

#loading the dataset into a dataframe

MultiOmicsDataset = pd.read_csv(r'Output/MultiOmicsDataset.csv')

#Delete first column as it does not mean anything
MultiOmicsDataset.drop(MultiOmicsDataset.columns[0],axis = 1, inplace = True)

#Shuffle the dataframe

MultiOmicsDataset = MultiOmicsDataset.sample(frac = 1)


#Separate the dataframe into train and test data, 75% goes to training and the 25% left goes to testing 

Train = MultiOmicsDataset.iloc[0:37473,:]
Test = MultiOmicsDataset.iloc[37473:,:]

print("Train and test data separated")
print("Train dataset's shape: " + str(Train.shape))
print("Test dataset's shape: " + str(Test.shape))

#Writing the datasets out to .csv files

print('-'*100)
print("Writing out the datasets to .csv files, please wait")
Train.to_csv(r'Output/TrainTestSamples/TrainDataset.csv')
Test.to_csv(r'Output/TrainTestSamples/TestDataset.csv')

print("Writing complete")
