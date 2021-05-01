#!usr/bin/python3 

#This is the genetic algorithm that will perform the feature selection, as presented in the report.

import numpy as np 

import random

import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.nn.functional as F

import deap
from deap import base, tools, creator

import pandas as pd

from tqdm import tqdm 

#----------------------------------------------------------------------
#CHECKING FOR GPU AVAILABILITY

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("Working on the GPU")
else:
  device = torch.device("cpu")
  print("Working on the CPU")

#-----------------------------------------------------------------------
#IMPORTING DATASETS

#Import Train and Test datasets
Train = pd.read_csv(r'TrainDataset.csv') 
Test = pd.read_csv(r'TestDataset.csv') 

#Delete first column as it is unnecessary 
Train.drop(Train.columns[0],axis = 1, inplace = True)
Test.drop(Test.columns[0],axis = 1, inplace = True)

#Convert the y dataframes to numpy arrays in order to use them as input for the model, the x dataframes are created when translating the feature vector in the function decodeChromosome() 
y_train2 = Train.iloc[:,700:].to_numpy()
y_test2 = Test.iloc[:,700:].to_numpy()

#The arrays above need to be converted from 2 classes to 1 to match the output of the network
y_train = []
y_test = []

for y in y_train2:
  y_train.append([np.where(y==1)[0][0]])  
  
for y in y_test2:
  y_test.append([np.where(y==1)[0][0]])  

#Convert to torch tensors
y_train = torch.as_tensor(y_train, dtype = torch.float32, device = device)
y_test = torch.as_tensor(y_test, dtype = torch.float32, device = device)





#-----------------------------------------------------------------------
#CREATING THE NETWORK CLASS 

#This is a raw version of the network class that will be used for prediction. It will suffer minor changes (for optimization purposes). 
class Network(nn.Module):

  # This constructor takes in the input size of the network as a parameter as, if the number of features to learn change, the input layer of the network needs to change as well)
  def __init__(self, inputSize):
    super(Network, self).__init__()
    self.hidden1 = nn.Linear(inputSize, 100)
    self.hidden2 = nn.Linear(100, 80)
    self.hidden3 = nn.Linear(80, 60)
    self.hidden4 = nn.Linear(60, 50)
    self.hidden5 = nn.Linear(50, 30)
    self.hidden6 = nn.Linear(30, 25)
    self.hidden7 = nn.Linear(25, 20)
    self.out = nn.Linear(20,1)


  # This function specifies the flow of information inside the network in each forward pass 
  def forward(self,x):
    x = F.relu(self.hidden1(x))
    x = F.relu(self.hidden2(x))
    x = F.relu(self.hidden3(x))
    x = F.relu(self.hidden4(x))
    x = F.relu(self.hidden5(x))
    x = F.relu(self.hidden6(x))
    x = F.relu(self.hidden7(x))
    output = torch.sigmoid(self.out(x))
    
    return output


#-----------------------------------------------------------------------
#SETTING UP THE EVOLUTIONARY ALGORITHM

#Initialising the hyperparameters for the evolutionary algorithm 
populationsize = 100 # Population size
ngenerations = 500 # Number of generations
crossprob = 0.7 # Crossover probability 
mutprob = 0.1 # Mutation probability
flipProb = 0.001 # Probability of a bit being flipped
nElitists = 1 # Number of elite individuals selected

#First we create the fitness atribute and the individual class 
creator.create("Fitness", base.Fitness, weights = (1.0,))
creator.create("Individual", list, fitness = creator.Fitness)

#Then we create the evolutionary tools and the toolbox to contain them 
toolbox = base.Toolbox()

toolbox.register("random", random.randint, 0,1)
toolbox.register("Individual", tools.initRepeat, creator.Individual, toolbox.random, 700)
toolbox.register("Population", tools.initRepeat, list, toolbox.Individual, populationsize)
toolbox.register("Selection", tools.selTournament, tournsize = 2, fit_attr = "fitness")
toolbox.register("Crossover", tools.cxTwoPoint)
toolbox.register("Mutate", tools.mutFlipBit, indpb = flipProb)




#-----------------------------------------------------------------------
#HELPER FUNCTIONS 

#This helper function decodes a chromosome (feature vector) and returns the elected columns of the dataframes converted to tensors (for training/testing purposes)

def decodeChromosome(FeatureVector, device):
  # create a copy of the training/testing dataframes  
  NewTrain = Train.iloc[:,:700].copy()
  NewTest = Test.iloc[:,:700].copy()
  

  IndNonElected = []
  # traverse the feature vector anotating the indexes not elected
  for i in range(len(FeatureVector)):
    if FeatureVector[i] == 0:
      IndNonElected.append(i)


  # drop the features not selected in the vector from both train and thest dataset
  NewTrain.drop( NewTrain.columns[IndNonElected] ,axis = 1, inplace = True)
  NewTest.drop( NewTest.columns[IndNonElected], axis = 1, inplace = True)

  # return both train and test datasets converted to torch tensors (stored in the GPU if possible)
  return torch.as_tensor(NewTrain.to_numpy(dtype="float32"), device = device), torch.as_tensor(NewTest.to_numpy(dtype = "float32"), device = device)


#This helper function takes an individual (feature vector of 1s and 0s) and counts the number of 1s inside (just for checking purposes)
def countOnes(individual):
  ones = 0 
  for i in individual:
    if i == 1:
      ones = ones +1

  return ones
  



#This helper function works almost the same way as the "decodeChromosome()" function, but it will be used to return the elected features as a dataframe

def returnDataframe(FeatureVector):
  # create a copy of the training/testing dataframes  
  NewTrain = Train.iloc[:,:700].copy()
  NewTest = Test.iloc[:,:700].copy()
  

  IndNonElected = []
  # traverse the feature vector anotating the indexes not elected
  for i in range(len(FeatureVector)):
    if FeatureVector[i] == 0:
      IndNonElected.append(i)


  # drop the features not selected in the vector from both train and thest dataset
  NewTrain.drop( NewTrain.columns[IndNonElected] ,axis = 1, inplace = True)
  NewTest.drop( NewTest.columns[IndNonElected], axis = 1, inplace = True)

  # return both train and test datasets converted to torch tensors (stored in the GPU if possible)
  return NewTrain, NewTest






#-----------------------------------------------------------------------
#MAIN METHOD

def main():
  
  # First, the population is created
  population = toolbox.Population()

  # Now, a list is created to keep track of the fitness of the best individual in each generation
  BestFitnesses = []

  currentInd = 0 
  # The first population created now can be evaluated
  for ind in population:
    currentInd = currentInd +1

    if currentInd % 5 == 0:
      print("Working on individual " + str(currentInd))
      torch.cuda.empty_cache()


    x_train, x_test = decodeChromosome(ind, device)

    # The network object is created as well as the optimizator that will help it learn
    net = Network(x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.02) 
  
    # The loss function is also initialised
    lossFunc = nn.BCELoss()

    # The network gets trained on the features selected for 200 iterations
    for i in range(200):
      optimizer.zero_grad()
      output = net(x_train)
      loss = lossFunc(output, y_train)
      loss.backward()
      
      optimizer.step()
      
    #Now, the model gets evaluated on the test dataset, and the inverse of its loss gets assigned as the individual's fitness 
    fitness = (1/lossFunc(net(x_test),y_test)).item() 
    ind.fitness.values = (fitness,)
    
    del net 
    del optimizer
    del x_train
    del x_test
    del lossFunc


  
  print("-"*30)
  print("Starting the evolutionary loop")
  # The evolutionary loop starts now
  for g in tqdm(range(ngenerations)):
  
    # Select the next generation of individuals
    offspring = tools.selBest(population,nElitists) + toolbox.Selection(population, len(population) - nElitists)

    # Clone the offspring to make sure we are working with a clean copy 
    offspring = list(map(toolbox.clone,offspring))

    print("performing the crossover")
    # Perform the crossover
    for child1, child2 in zip(offspring[::2],offspring[1::2]):
      if random.random() < crossprob:
        toolbox.Crossover(child1,child2)
        del child1.fitness.values
        del child2.fitness.values


    print("performing the mutation")
    # Perform the mutation 
    for mutant in offspring:
      if random.random() < mutprob:
        toolbox.Mutate(mutant)
        del mutant.fitness.values

    
    print("performing the evaluation")
    currentInd = 0 
    # Evaluate the new population
    for ind in offspring:
      
      currentInd = currentInd +1

      if currentInd % 5 == 0:
        print("Working on individual " + str(currentInd))
        torch.cuda.empty_cache()

      x_train, x_test = decodeChromosome(ind, device) 
      # The network object is created as well as the optimizator that will help it learn
      net = Network(x_train.shape[1]).to(device)
      optimizer = torch.optim.Adam(net.parameters(), lr = 0.02) 
      # The loss function is also initialised
      lossFunc = nn.BCELoss()

      # The network gets trained on the features selected for 200 iterations
      for i in range(200):
        optimizer.zero_grad()
        output = net(x_train)
        loss = lossFunc(output, y_train)
        loss.backward()
      
        optimizer.step()
      
      #Now, the model gets evaluated on the test dataset, and the inverse of its loss gets assigned as the individual's fitness 
      fitness = (1/lossFunc(net(x_test),y_test)).item() 
      ind.fitness.values = (fitness,)
      
      del net
      del optimizer
      del x_train
      del x_test
      del lossFunc
       
    # Now the offspring population becomes the parent population 
    population[:] = offspring

    


    # In order to achieve the best combination of features, we need to keep track of the best individual across all populations

    # Fist get the best individual of the current generation
    BestOfGeneration = tools.selBest(population,1)[0]
    # Set it as the global best in the first generation
    if g == 0: 
      currentBest = BestOfGeneration 
    else:
      # For the rest of generations, compare the local best with the global best
      if BestOfGeneration.fitness.values[0] > currentBest.fitness.values[0]:
        currentBest = BestOfGeneration 

    BestFitnesses.append(BestOfGeneration.fitness.values[0])

    print("Best fitness of current generation: " + str(BestOfGeneration.fitness.values[0]))
    

  print("-"*30)
  print("Evolution complete")
  
  # Save the best combination of features into a new txt file 
  FinalTrain, FinalTest = returnDataframe(currentBest)
  FinalTrain.to_csv(r'FinalTrainDataset.csv')
  FinalTest.to_csv(r'FinalTestDataset.csv')
  
  # Print the best fitnesses accross generations
  print("Best fitnesses accross generations")
  print(BestFitnesses)

  #Print best fitness of all 
  print("Best fitness")
  print(currentBest)

  # Plot the best fitnesses accross generations
  plt.title("Best fitnesses across generations")
  plt.plot(BestFitnesses)
  plt.xlabel("Generations")
  plt.ylabel("Best fitness")
  plt.savefig('BestFitnesses.png')

if __name__ == "__main__":
  main()
