#%% Imports
    
import time

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import tree

import torch
import torch.nn as nn                                        
import torch.nn.functional as F   

import matplotlib.pyplot as plt

#%% Data processing
data_directory = "D:/git/mlp_pinn/"

# Load X
generator_dispatch = np.loadtxt(open(data_directory + "generator_dispatch.csv", "rb"), delimiter=",", skiprows=0).T
print("Generator dispatch data loaded, transposed.")

# Load y
classification_N1 = np.loadtxt(open(data_directory + "classification_N1.csv", "rb"), delimiter=",", skiprows=0)
classification_N1 = np.row_stack(([classification_N1], [np.ones(classification_N1.shape) - classification_N1])).T
classification_N1_SSS = np.loadtxt(open(data_directory + "classification_N1_SSS.csv", "rb"), delimiter=",", skiprows=0)
classification_N1_SSS = np.row_stack(([classification_N1_SSS], [np.ones(classification_N1_SSS.shape) - classification_N1_SSS])).T
print("Security classification data loaded, transposed, one-hot encoded.")

X_N1_train, X_N1_test, y_N1_train, y_N1_test = train_test_split(generator_dispatch, classification_N1, test_size=0.2)
X_N1_SSS_train, X_N1_SSS_test, y_N1_SSS_train, y_N1_SSS_test = train_test_split(generator_dispatch, classification_N1_SSS, test_size=0.2)

#%% Performance metric function definitions 

def compute_accuracy(y, yp, X):
    """Compute and return the accuracy of prediction yp based on dataset X and classification y"""
    return 100 * (1 - np.sum(np.absolute(yp - y)) / (len(X[:, 0])))

def compute_mcc(y, yp, X):
    """Compute and return the Matthews correlation coefficient of prediction yp"""
    if len(y.shape) == 2: # if one-hot encoded, convert back
        y = y[:, 0]
        yp = yp[:, 0]
    
    tp = np.sum((y == 1) & (yp == 1), dtype='int64')
    tn = np.sum((y == 0) & (yp == 0), dtype='int64')
    fp = np.sum((y == 0) & (yp == 1), dtype='int64')
    fn = np.sum((y == 1) & (yp == 0), dtype='int64')
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return mcc

#%% Modeling - decision tree
print("-------------------------------------------")
print("N1 - Training a decision tree and evaluating it.")
tree_N1 = tree.DecisionTreeClassifier()
tree_N1 = tree_N1.fit(X_N1_train, y_N1_train[:, 0])

plt.figure()
tree.plot_tree(tree_N1)

yp_N1_train = tree_N1.predict(X_N1_train)
yp_N1_test = tree_N1.predict(X_N1_test)

acc_train = compute_accuracy(y_N1_train[:, 0], yp_N1_train, X_N1_train)
acc_test = compute_accuracy(y_N1_test[:, 0], yp_N1_test, X_N1_test)
print(f"Decision tree training accuracy: {acc_train}%")
print(f"Decision tree test accuracy {round(acc_test, 2)}%")
mcc_train = compute_mcc(y_N1_train[:, 0], yp_N1_train, X_N1_train)
mcc_test = compute_mcc(y_N1_test[:, 0], yp_N1_test, X_N1_test)
print(f"Decision tree training MCC: {mcc_train}")
print(f"Decision tree test MCC {round(mcc_test, 2)}")

print("-------------------------------------------")
print("N1_SSS - Training a decision tree and evaluating it.")
tree_N1_SSS = tree.DecisionTreeClassifier()
tree_N1_SSS = tree_N1.fit(X_N1_SSS_train, y_N1_SSS_train[:, 0])
tree.plot_tree(tree_N1_SSS)

yp_N1_SSS_train = tree_N1_SSS.predict(X_N1_SSS_train)
yp_N1_SSS_test = tree_N1_SSS.predict(X_N1_SSS_test)

acc_train = compute_accuracy(y_N1_SSS_train[:, 0], yp_N1_SSS_train, X_N1_SSS_train)
acc_test = compute_accuracy(y_N1_SSS_test[:, 0], yp_N1_SSS_test, X_N1_SSS_test)
print(f"Decision tree training accuracy: {acc_train}%")
print(f"Decision tree test accuracy {round(acc_test, 2)}%")
mcc_train = compute_mcc(y_N1_SSS_train[:, 0], yp_N1_SSS_train, X_N1_SSS_train)
mcc_test = compute_mcc(y_N1_SSS_test[:, 0], yp_N1_SSS_test, X_N1_SSS_test)
print(f"Decision tree training MCC: {mcc_train}")
print(f"Decision tree test MCC {round(mcc_test, 2)}")

#%% Modeling - neural network
print("-------------------------------------------")
print("N1 - Training a neural network and evaluating it.")

try: # F.Linear struggles with float64. In a 'try' block for quick running of this cell.
    X_N1_train = torch.from_numpy(X_N1_train).to(torch.float32) 
    y_N1_train = torch.from_numpy(y_N1_train).to(torch.float32) 
    X_N1_test = torch.from_numpy(X_N1_test).to(torch.float32) 
    y_N1_test = torch.from_numpy(y_N1_test).to(torch.float32) 
except:
    print("Failed to convert data to usable PyTorch tensors.")

# Model set-up - sometimes the NN fails to train, assumed to be due to
# the specific hyperparameter selection.
n_in = len(X_N1_train[0, :])
n_out = len(y_N1_train[0, :])
n_hidden = 30
learning_rate = 0.005 # Do not go above 0.01

NN = nn.Sequential(nn.Linear(n_in, n_hidden), # Input layer.
                   nn.ReLU(),
                   nn.Linear(n_hidden, n_hidden), # First hidden layer
                   nn.ReLU(),
                   nn.Linear(n_hidden, n_hidden), # Second hidden layer
                   nn.ReLU(),
                   nn.Linear(n_hidden, n_hidden), # Third hidden layer
                   nn.ReLU(),
                   nn.Linear(n_hidden, n_out),
                   nn.Softmax(dim=1)) # This dim seems to be right.

loss_function = nn.BCELoss()
optimizer = torch.optim.SGD(NN.parameters(), lr=learning_rate)

n_epochs = 1000
min_n_epochs = 50


timer_start = time.time()
losses = []
for epoch in range(n_epochs + 1):
    if epoch % 100 == 0:
        print(f"Epoch #: {epoch}")
    yp_N1_train = NN(X_N1_train)
    loss = loss_function(yp_N1_train, y_N1_train)
    
    if epoch > min_n_epochs: # Early stop - stop training if no further improvement.
        if (round(loss.item(), 5) == round(np.average(losses[-10]), 5)):
                                   print(f"Ended early at epoch #: {epoch}.")
                                   break
                               
    losses.append(loss.item())
    
    NN.zero_grad()
    loss.backward()
    
    optimizer.step()

timer_end = time.time()
print(f"NN trained in {round(timer_end - timer_start)}s.")
yp_N1_test = NN(X_N1_test)

acc_train = compute_accuracy(y_N1_train.numpy(), yp_N1_train.detach().numpy(), X_N1_train)
acc_test = compute_accuracy(y_N1_test.numpy(), yp_N1_test.detach().numpy(), X_N1_test)
print(f"NN training accuracy: {round(acc_train, 2)}%")
print(f"NN test accuracy {round(acc_test, 2)}%")
# WARNING: We introduce a probability treshold in the classification by computing 
#          MCC based on np.round (assumed to be 0.5)
mcc_train = compute_mcc(y_N1_train.numpy(), np.round(yp_N1_train.detach().numpy(), 0), X_N1_train)
mcc_test = compute_mcc(y_N1_test.numpy(), np.round(yp_N1_test.detach().numpy(), 0), X_N1_test)
print(f"NN training MCC: {round(mcc_train, 2)}")
print(f"NN test MCC {round(mcc_test, 2)}")
# INSIGHT: Accuracy may be above 95%, but the MCC can still be dogshit (0.35-0.70).
# 3.1.5 This is due to the data not being balanced, and accuracy thus not being 
# a good metric for this scenario.

plt.figure(figsize=[8, 6])
plt.plot(losses)
plt.ylabel('Loss [?]')
plt.xlabel('Epoch # [-]')
plt.show()