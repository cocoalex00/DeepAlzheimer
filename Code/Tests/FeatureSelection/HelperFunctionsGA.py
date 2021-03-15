#!usr/bin/python3


#In order to keep the code clear, This file will contain the helper functions needed for the GA performing the feature selection

import numpy as np 
import random
import torch 
import deap
from deap import base, tools, creator
import pandas as pd

#Import Train and Test datasets
Train = pd.read_csv(r'TrainDataset.csv') 
Test = pd.read_csv(r'TestDataset.csv') 
#Delete first column as it is unnecessary 

Train.drop(Train.columns[0],axis = 1, inplace = True)
Test.drop(Test.columns[0],axis = 1, inplace = True)
#This helper function decodes a chromosome (feature vector) and returns the elected columns of the dataframes converted to numpy arrays (for training/testing purposes)

def decodeChromosome(FeatureVector):
  #create a copy of the training/testing dataframes  
  NewTrain = Train.iloc[:,:700].copy()
  NewTest = Test.iloc[:,:700].copy()
  

  IndNonElected = []
  #traverse the feature vector anotating the indexes not elected
  for i in range(len(FeatureVector) - 1):
    if FeatureVector[i] == 0:
      IndNonElected.append(i)

  #drop the features not selected in the vector from both train and thest dataset
  NewTrain.drop( NewTrain.columns[IndNonElected] ,axis = 1, inplace = True)
  NewTest.drop( NewTest.columns[IndNonElected], axis = 1, inplace = True)
  return NewTrain.to_numpy(), NewTest.to_numpy()



    
  
  

#Main method
def main():
  
  ind = toolbox.Individual()
  print(ind)
  x_train, x_test, = decodeChromosome(ind)
  print(x_test.shape)

if __name__ == "__main__":
  main()

  
#print(x_train)
#print(y_train)
#print(x_test)
#print(y_test)
#print(Test)
