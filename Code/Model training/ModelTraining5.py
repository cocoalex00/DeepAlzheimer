#!usr/bin/python3

#This file will perform the training of a selected model as presented in the report 

import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F

import pandas as pd

#-----------------------------------------------------------------------
#CHECKING FOR GPU AVAILABILITY

if torch.cuda.is_available():
  device = torch.device("cuda:0")
  print("Working on the GPU")
else:
  device = torch.device("cpu")
  print("Working on the CPU")


#----------------------------------------------------------------------
#IMPORTING ALL DATASETS   

# Load multi-omics dataset (To extract the labels of the samples)
MultiOmicsTrain = pd.read_csv(r'Datasets/TrainDataset.csv')
MultiOmicsTest = pd.read_csv(r'Datasets/TestDataset.csv')

# Load train and test dataset selected in the first run of the feature selection 
TrainFirstRun = pd.read_csv(r'Datasets/FinalTrainDatasetR1.csv')
TestFirstRun = pd.read_csv(r'Datasets/FinalTestDatasetR1.csv')

# Load train and test dataset selected in the second run of the feature selection 
TrainSecondRun = pd.read_csv(r'Datasets/FinalTrainDatasetR2.csv')
TestSecondRun = pd.read_csv(r'Datasets/FinalTestDatasetR2.csv')

# Load train and test dataset selected in the third run of the feature selection 
TrainThirdRun = pd.read_csv(r'Datasets/FinalTrainDatasetR3.csv')
TestThirdRun = pd.read_csv(r'Datasets/FinalTestDatasetR3.csv')

print('-'*30)
print("Datasets loaded")

# Delete the first column as it is unnecessary 
MultiOmicsTrain.drop(MultiOmicsTrain.columns[0], axis = 1, inplace = True)
MultiOmicsTest.drop(MultiOmicsTest.columns[0], axis = 1, inplace = True)

TrainFirstRun.drop(TrainFirstRun.columns[0], axis = 1, inplace = True)
TestFirstRun.drop(TestFirstRun.columns[0], axis = 1, inplace = True)

TrainSecondRun.drop(TrainSecondRun.columns[0], axis = 1, inplace = True)
TestSecondRun.drop(TestSecondRun.columns[0], axis = 1, inplace = True)

TrainThirdRun.drop(TrainThirdRun.columns[0], axis = 1, inplace = True)
TestThirdRun.drop(TestThirdRun.columns[0], axis = 1, inplace = True)

# Convert the datasets to tensors

TrainFirstRun = torch.as_tensor(TrainFirstRun.to_numpy(), dtype = torch.float32, device = device)
TestFirstRun = torch.as_tensor(TestFirstRun.to_numpy(), dtype = torch.float32, device = device)

TrainSecondRun = torch.as_tensor(TrainSecondRun.to_numpy(), dtype = torch.float32, device = device)
TestSecondRun = torch.as_tensor(TestSecondRun.to_numpy(), dtype = torch.float32, device = device)

TrainThirdRun = torch.as_tensor(TrainThirdRun.to_numpy(), dtype = torch.float32, device = device)
TestThirdRun = torch.as_tensor(TestThirdRun.to_numpy(), dtype = torch.float32, device = device)


# Convert the y dataframes to numpy arrays in order to use them as input for the model

y_train2 = MultiOmicsTrain.iloc[:,700:].to_numpy()
y_test2 = MultiOmicsTest.iloc[:,700:].to_numpy()

# The arrays above need to be converted from 2 classes to 1 to match the output of the network
y_train = []
y_test = []

for y in y_train2:
  y_train.append([np.where(y==1)[0][0]])

for y in y_test2:
  y_test.append([np.where(y==1)[0][0]])

# Convert the label arrays to torch tensors
y_train = torch.as_tensor(y_train, dtype = torch.float32, device = device)
y_test = torch.as_tensor(y_test, dtype = torch.float32, device = device)

print('-'*30)
print("Datasets processed and converted to tensors")



#----------------------------------------------------------------------o#CREATING THE NETWORK CLASS

class Network(nn.Module):

  def __init__(self, inputSize):
    super(Network, self).__init__()
    self.hidden1 = nn.Linear(inputSize, 100)
    self.hidden2 = nn.Linear(100,80)
    self.hidden3 = nn.Linear(80,60)
    self.hidden4 = nn.Linear(60,50)
    self.hidden5 = nn.Linear(50, 30)
    self.out = nn.Linear(30, 1)

  
  def forward(self,x):
    x = F.relu(self.hidden1(x))
    x = F.relu(self.hidden2(x))
    x = F.relu(self.hidden3(x))
    x = F.relu(self.hidden4(x))
    x = F.relu(self.hidden5(x))
    x = torch.sigmoid(self.out(x))

    return x
    

#-----------------------------------------------------------------------#MAIN METHOD

def main():
  print("hello world")




if __name__ == "__main__":
  main()
