#!usr/bin/python3

#This file will perform the training of a selected model as presented in the report 

from sklearn.metrics import accuracy_score , roc_auc_score
import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F

import pandas as pd

import matplotlib.pyplot as plt 

from tqdm import tqdm

from sklearn.metrics import confusion_matrix, precision_score
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
MultiOmicsTrain = pd.read_csv(r'Datasets/MultiOmicsTrain.csv')
MultiOmicsTest = pd.read_csv(r'Datasets/MultiOmicsTest.csv')

# Load train and test dataset selected in the first run of the feature selection 
TrainFirstRun = pd.read_csv(r'Datasets/FinalTrainDataset5.csv')
TestFirstRun = pd.read_csv(r'Datasets/FinalTestDataset5.csv')

# Load train and test dataset selected in the second run of the feature selection 
TrainSecondRun = pd.read_csv(r'Datasets/FinalTrainDataset6.csv')
TestSecondRun = pd.read_csv(r'Datasets/FinalTestDataset6.csv')

# Load train and test dataset selected in the third run of the feature selection 
TrainThirdRun = pd.read_csv(r'Datasets/FinalTrainDataset7.csv')
TestThirdRun = pd.read_csv(r'Datasets/FinalTestDataset7.csv')

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

# Drop the columns with label information from the Multi Omics dataset 
MultiOmicsTrain.drop(MultiOmicsTrain.columns[700:], axis = 1, inplace = True)
MultiOmicsTest.drop(MultiOmicsTest.columns[700:], axis = 1, inplace = True)


# Convert the full multi omics dataset to tensors for training and testing
MultiOmicsTrain = torch.as_tensor(MultiOmicsTrain.to_numpy(), dtype = torch.float32, device = device)
MultiOmicsTest = torch.as_tensor(MultiOmicsTest.to_numpy(), dtype = torch.float32, device = device)

print('-'*30)
print("Datasets processed and converted to tensors")

#----------------------------------------------------------------------
#HELPER FUNCTIONS

# This helper function takes an output vector and returns the logits predicted by the model 
def returnLogits(output):
  logits = []

  for i in output:
    if i[0] < 0.5:
      logits.append([0])
    elif i[0] > 0.5:
      logits.append([1])
  return np.array(logits)

# This helper function takes 4 models and gives the user the oportunity to save their weights and biases to external files for later access 
def saveModels(recurrent,net1,net2,net3,net4):
    
  if recurrent == False:
      decision = input("Do you wish to save these models' state dictionaries?(y/n) : ")

      if decision == "y":
        torch.save(net1.state_dict(), 'ModelWeights/model1.pt')
        torch.save(net2.state_dict(), 'ModelWeights/model2.pt')
        torch.save(net3.state_dict(), 'ModelWeights/model3.pt')
        torch.save(net4.state_dict(), 'ModelWeights/model4.pt')
      elif decision == "n":
        print("no model was saved")
      else:
        saveModels(True)
  else:
      decision = input("The input was not recognized, please type y or n: ")

      if decision == "y":
        torch.save(net1.state_dict(), 'ModelWeights/model1.pt')
        torch.save(net2.state_dict(), 'ModelWeights/model2.pt')
        torch.save(net3.state_dict(), 'ModelWeights/model3.pt')
        torch.save(net4.state_dict(), 'ModelWeights/model4.pt')
      elif decision == "n":
        print("no model was saved")
      else:
        saveModels(True)
    
  
    
#----------------------------------------------------------------------
#CREATING THE NETWORK CLASS

class Network(nn.Module):

  def __init__(self, inputSize):
    super(Network, self).__init__()
    self.hidden1 = nn.Linear(inputSize, 100)
    self.hidden2 = nn.Linear(100,80)
    self.hidden3 = nn.Linear(80,60)
    self.hidden4 = nn.Linear(60,50)
    self.hidden5 = nn.Linear(50, 30)
    self.hidden6 = nn.Linear(30, 25)
    self.hidden7 = nn.Linear(25, 15)
    self.out = nn.Linear(15, 1)
    
    self.dropout = nn.Dropout(0.5)
  
  def forward(self,x):
    x = F.relu(self.hidden1(x))
    x = F.relu(self.hidden2(x))
    #x = self.dropout(x)
    x = F.relu(self.hidden3(x))
    x = F.relu(self.hidden4(x))
    x = self.dropout(x)
    x = F.relu(self.hidden5(x))
    x = self.dropout(x)
    x = F.relu(self.hidden6(x))
    x = F.relu(self.hidden7(x))
    x = torch.sigmoid(self.out(x))

    return x
    

#-----------------------------------------------------------------------
#MAIN METHOD

def main():
  
  #Initialise seed for reproducibility
  torch.manual_seed(89)

  # initialise hyperparmeters and network objects
  lr = 0.0001
  nOfIterations = 435
  netRaw = Network(MultiOmicsTest.shape[1]).to(device)
  net1 = Network(TestFirstRun.shape[1]).to(device)
  net2 = Network(TestSecondRun.shape[1]).to(device)
  net3 = Network(TestThirdRun.shape[1]).to(device)
  
  # Set the networks in train mode (to activate dropout)
  netRaw.train()
  net1.train()
  net2.train()
  net3.train()
  
  # Initialise optimizers and loss function

  optimizerRaw = torch.optim.Adam(netRaw.parameters(), lr = lr)
  optimizer1 = torch.optim.Adam(net1.parameters(), lr = lr)
  optimizer2 = torch.optim.Adam(net2.parameters(), lr = lr)
  optimizer3 = torch.optim.Adam(net3.parameters(), lr = lr)

  lossFuncRaw = nn.BCELoss()
  lossFunc1 = nn.BCELoss()
  lossFunc2 = nn.BCELoss()
  lossFunc3 = nn.BCELoss()

  # Initialise arrays to keep track of loss
  LossesRaw = []
  Losses1 = []
  Losses2 = []
  Losses3 = []
  

  # Training loop 
  for i in tqdm(range(nOfIterations)):
    # Zero the gradients of all optimizers
    optimizerRaw.zero_grad()
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    optimizer3.zero_grad()
    
    # Calculate the output of the networks 
    outputRaw = netRaw(MultiOmicsTrain)
    output1 = net1(TrainFirstRun)
    output2 = net2(TrainSecondRun)
    output3 = net3(TrainThirdRun)

    # Calculate the loss of all networks
    lossRaw = lossFuncRaw(outputRaw, y_train)
    loss1 = lossFunc1(output1, y_train)
    loss2 = lossFunc2(output2, y_train)
    loss3 = lossFunc3(output3, y_train)

    # Append the loss to the arrays (for visualization purposes)
    LossesRaw.append(lossRaw)
    Losses1.append(loss1)
    Losses2.append(loss2)
    Losses3.append(loss3)
   
    # Calculate the gradients
    lossRaw.backward()
    loss1.backward()
    loss2.backward()
    loss3.backward()

    # Take an optimization step
    optimizerRaw.step()
    optimizer1.step()
    optimizer2.step() 
    optimizer3.step()



  plt.plot(Losses1, label ='Loss (dataset 1)')
  plt.plot(Losses2,'r', label = 'Loss (dataset 2)')
  plt.plot(Losses3, 'y', label = 'Loss (dataset3)')
  plt.plot(LossesRaw, 'g', label = 'Loss (Multi-omics dataset)')
  plt.xlabel("iteration")
  plt.ylabel("loss")
  plt.legend()
  plt.show()

  # Set the networks in eval mode (to deactivate dropout)
  netRaw.eval()
  net1.eval()
  net2.eval()
  net3.eval()
  
  testOutput1 = net1(TestFirstRun)
  testOutput2 = net2(TestSecondRun)
  testOutput3 = net3(TestThirdRun)
  testOutputRaw = netRaw(MultiOmicsTest)
    
  logits1 = returnLogits(testOutput1)
  logits2 = returnLogits(testOutput2)
  logits3 = returnLogits(testOutput3)
  logitsRaw = returnLogits(testOutputRaw)
    
    
  accuracy1 = accuracy_score(y_test,logits1)
  accuracy2 = accuracy_score(y_test,logits2)
  accuracy3 = accuracy_score(y_test,logits3)
  accuracyRaw = accuracy_score(y_test,logitsRaw)


  auroc1 = roc_auc_score(y_test,logits1)
  auroc2 = roc_auc_score(y_test,logits2)
  auroc3 = roc_auc_score(y_test,logits3)
  aurocRaw = roc_auc_score(y_test,logitsRaw)
    
  cm1 = confusion_matrix(y_test,logits1)
  cm2 = confusion_matrix(y_test,logits2)
  cm3 = confusion_matrix(y_test,logits3)
  cmraw = confusion_matrix(y_test,logitsRaw)
    
  Ps1 = precision_score(y_test,logits1)
  Ps2 = precision_score(y_test,logits2)
  Ps3 = precision_score(y_test,logits3)
  Psraw = precision_score(y_test,logitsRaw)

  IPs1 = precision_score(np.logical_not(y_test).astype(int),np.logical_not(logits1).astype(int))
  IPs2 = precision_score(np.logical_not(y_test).astype(int),np.logical_not(logits2).astype(int))
  IPs3 = precision_score(np.logical_not(y_test).astype(int),np.logical_not(logits3).astype(int))
  IPsraw = precision_score(np.logical_not(y_test).astype(int),np.logical_not(logitsRaw).astype(int))

  print("accuracy of model 1: " + str(accuracy1))
  print("accuracy of model 2: " + str(accuracy2))
  print("accuracy of model 3: " + str(accuracy3))
  print("accuracy of model 4 (dataset without FS): " + str(accuracyRaw))
  print('-'*30)
  
  print("Auroc of model 1: " + str(auroc1))
  print("Auroc of model 2: " + str(auroc2))
  print("Auroc of model 3: " + str(auroc3))
  print("Auroc of model 4 (dataset without FS): " + str(aurocRaw))
  print('-'*30)

  print("Confusion Matrix of model 1: ")
  print(str(cm1))
  print("Confusion Matrix of model 2: ")
  print(str(cm2))
  print("Confusion Matrix of model 3: ")
  print(str(cm3))
  print("Confusion Matrix of model 4 (dataset without FS): ")
  print(str(cmraw))
  print('-'*30)
    
  print("Precision of model 1: " + str(Ps1))
  print("Precision of model 2: " + str(Ps2))
  print("Precision of model 3: " + str(Ps3))
  print("Precision of model 4 (dataset without FS): " + str(Psraw))
  print('-'*30) 

  print("Precision of model 1 (For non AD): " + str(IPs1))
  print("Precision of model 2 (For non AD): " + str(IPs2))
  print("Precision of model 3 (For non AD): " + str(IPs3))
  print("Precision of model 4 (dataset without FS) (For non AD): " + str(IPsraw))

  print('-'*30) 
    
  # Give the oportunity of saving the models' states for later usage 
  saveModels(False,net1,net2,net3,netRaw)

if __name__ == "__main__":
  main()
