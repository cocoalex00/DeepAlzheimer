#!usr/bin/python3

# This code will fit a logistic regresion model on the three datasets in order to compare its performance with the proposed models

from sklearn.metrics import accuracy_score , roc_auc_score
import numpy as np 

import sklearn 
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

import matplotlib.pyplot as plt 

from tqdm import tqdm

from sklearn.metrics import confusion_matrix, precision_score


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
TrainFirstRun = TrainFirstRun.to_numpy()
TestFirstRun = TestFirstRun.to_numpy()

TrainSecondRun = TrainSecondRun.to_numpy()
TestSecondRun = TestSecondRun.to_numpy()
TrainThirdRun = TrainThirdRun.to_numpy()
TestThirdRun = TestThirdRun.to_numpy()

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


# Drop the columns with label information from the Multi Omics dataset 
MultiOmicsTrain.drop(MultiOmicsTrain.columns[700:], axis = 1, inplace = True)
MultiOmicsTest.drop(MultiOmicsTest.columns[700:], axis = 1, inplace = True)


# Convert the full multi omics dataset to tensors for training and testing
MultiOmicsTrain = MultiOmicsTrain.to_numpy()
MultiOmicsTest = MultiOmicsTest.to_numpy()
print('-'*30)
print("Datasets processed and converted to tensors")


def main():
  

  # Initialise 4 classifiers and fit them to the train data 
  clf1 = KNeighborsClassifier().fit(TrainFirstRun,y_train)
  clf2 = KNeighborsClassifier().fit(TrainSecondRun,y_train)
  clf3 = KNeighborsClassifier().fit(TrainThirdRun,y_train)
  clfRaw = KNeighborsClassifier().fit(MultiOmicsTrain,y_train)

  # Predict the outputs of the test dataset
  testOutput1 = clf1.predict(TestFirstRun)
  testOutput2 = clf2.predict(TestSecondRun)
  testOutput3 = clf3.predict(TestThirdRun)
  testOutputRaw = clfRaw.predict(MultiOmicsTest)
  
  # Calculate the accuracy of all classifiers
  accuracy1 = accuracy_score(y_test,testOutput1)
  accuracy2 = accuracy_score(y_test,testOutput2)
  accuracy3 = accuracy_score(y_test,testOutput3)
  accuracyRaw = accuracy_score(y_test,testOutputRaw)

  # Calculate the AUC of all classifiers
  auroc1 = roc_auc_score(y_test,testOutput1)
  auroc2 = roc_auc_score(y_test,testOutput2)
  auroc3 = roc_auc_score(y_test,testOutput3)
  aurocRaw = roc_auc_score(y_test,testOutputRaw)
    
  # Calculate the confusion matrix of all classifiers    
  cm1 = confusion_matrix(y_test,testOutput1)
  cm2 = confusion_matrix(y_test,testOutput2)
  cm3 = confusion_matrix(y_test,testOutput3)
  cmraw = confusion_matrix(y_test,testOutputRaw)

  # Calculate the precision of all classifiers (AD)   
  Ps1 = precision_score(y_test,testOutput1)
  Ps2 = precision_score(y_test,testOutput2)
  Ps3 = precision_score(y_test,testOutput3)
  Psraw = precision_score(y_test,testOutputRaw)

  # Calculate the precision of all classifiers (Non-AD)
  IPs1 = precision_score(np.logical_not(y_test).astype(int),np.logical_not(testOutput1).astype(int))
  IPs2 = precision_score(np.logical_not(y_test).astype(int),np.logical_not(testOutput2).astype(int))
  IPs3 = precision_score(np.logical_not(y_test).astype(int),np.logical_not(testOutput3).astype(int))
  IPsraw = precision_score(np.logical_not(y_test).astype(int),np.logical_not(testOutputRaw).astype(int))

  #Print all metrics
    
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

if __name__ == "__main__":
  main()
