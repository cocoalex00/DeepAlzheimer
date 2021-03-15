#!usr/bin/python3

#This is a test implementation of the Genetic algorithm that will perform the feature selection. In this implementation, only the training of the models will be performed in the GPU, the GA itself won't be parallel as that will be implemented based on this verion.

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

#Convert the y dataframes to numpy arrays in order to use them as input for the model, the x dataframes are created when translating the feature vector in the function decodeChromosome() 
y_train = Train.iloc[:,700:].to_numpy()
y_test = Test.iloc[:,700:].to_numpy()





#-----------------------------------------------------------------------
#SETTING UP THE EVOLUTIONARY ALGORITHM

#Initialising the hyperparameters for the evolutionary algorithm 
populationsize = 100 # Population size
ngenerations = 20 # Number of generations
crossprob = 0.7 # Crossover probability 
mutprob = 0.80 # Mutation probability



#First we create the fitness atribute and the individual class 
creator.create("Fitness", base.Fitness, weights = (1.0,))
creator.create("Individual", list, fitness = creator.Fitness)

#Then we create the evolutionary tools and the toolbox to contain them 
toolbox = base.Toolbox()

toolbox.register("random", random.randint, 0,1)
toolbox.register("Individual", tools.initRepeat, creator.Individual, toolbox.random, 700)
toolbox.register("Population", tools.initRepeat, list, creator.Individual, populationsize)
toolbox.register("Selection", tools.selTournament, tournsize = 2, fit_attr = "fitness")
toolbox.register("Crossover", tools.cxUniform, indpb = 0.5)
toolbox.register("Mutate", tools.mutFlipBit, indpb = 0.05)







#-----------------------------------------------------------------------
#HELPER FUNCTIONS 

#This helper function decodes a chromosome (feature vector) and returns the elected columns of the dataframes converted to tensors (for training/testing purposes)

def decodeChromosome(FeatureVector):
  # create a copy of the training/testing dataframes  
  NewTrain = Train.iloc[:,:700].copy()
  NewTest = Test.iloc[:,:700].copy()
  

  IndNonElected = []
  # traverse the feature vector anotating the indexes not elected
  for i in range(len(FeatureVector) - 1):
    if FeatureVector[i] == 0:
      IndNonElected.append(i)

  # drop the features not selected in the vector from both train and thest dataset
  NewTrain.drop( NewTrain.columns[IndNonElected] ,axis = 1, inplace = True)
  NewTest.drop( NewTest.columns[IndNonElected], axis = 1, inplace = True)

  
  return torch.as_tensor(NewTrain.to_numpy()), torch.as_tensor(NewTest.to_numpy())



    
  
  












#-----------------------------------------------------------------------
#MAIN METHOD

def main():
  
  ind = toolbox.Individual()
  print(ind)
  x_train, x_test, = decodeChromosome(ind)
  print(x_test.shape)

if __name__ == "__main__":
  main()

  
