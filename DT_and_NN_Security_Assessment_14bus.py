# DISCLAIMER: Most of this code is provided as an assignment, please do not 
#             attribute the lack of coding style conventions to me (Dmitrij)

# all these modules are necessary
import numpy # as np, this hurts, who would just do this
from sklearn import tree
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# insert the directory where the data is stored
data_directory = "D:/git/mlp_pinn/"

# COMMENT - loads X, OPF database without VG
print("Loading generator set-points -- features for machine learning")
# ATTENTION: The MAT files need to be converted to CSV files to be imported (see MATLAB fct for that)
generator_dispatch = numpy.loadtxt(open(data_directory + "generator_dispatch.csv", "rb"), delimiter=",", skiprows=0)
print(generator_dispatch)
print(generator_dispatch.shape)
# normalize features for better performance

# COMMENT - loads y
print("Loading security classification N-1 security")
classification_N1 = numpy.loadtxt(open(data_directory + "classification_N1.csv", "rb"), delimiter=",", skiprows=0)
# COMMENT - one hot encode (OHE) the binary classification y vector (for improved NN operation), y_1 = 1 safe, y_2 = 2 unsafe
classification_N1 = numpy.row_stack(([classification_N1], [numpy.ones(classification_N1.shape) - classification_N1]))
print(classification_N1)
print(classification_N1.shape)
print("The number of secure samples:", numpy.sum(classification_N1[0])) # 885
# assignment assumed to be missing the [0], otherwise it just gives you the number of cases due to OHE
# not following the database convention of examples stored in rows, this hurts 

# COMMENT - loads y, with a different (more strict) classification criterion
print("security classification N-1 security and small-signal stability")
classification_N1_SSS = numpy.loadtxt(open(data_directory + "classification_N1_SSS.csv", "rb"), delimiter=",",
                                      skiprows=0)
# COMMENT - OHE the binary classification y vector 
classification_N1_SSS = numpy.row_stack(
    ([classification_N1_SSS], [numpy.ones(classification_N1_SSS.shape) - classification_N1_SSS]))
print(classification_N1_SSS)
print(numpy.sum(classification_N1))
print("The number of secure samples:", numpy.sum(classification_N1_SSS[0])) # 218
# assignment assumed to be missing the [0], otherwise it just gives you the number of cases due to OHE

# COMMENT - shuffle the datasets, split into training/test (0.8/0.2) sets
print(classification_N1.shape[0])
indices = numpy.random.permutation(generator_dispatch.shape[1])
print(indices)
nr_samples = len(indices)
print(nr_samples)
size_train = int(numpy.rint(0.8 * nr_samples))
print(size_train)
training_idx, test_idx = indices[:size_train], indices[size_train:]

generator_dispatch_train = generator_dispatch[:, training_idx]
generator_dispatch_test = generator_dispatch[:, test_idx]

classification_N1_train = classification_N1[:, training_idx]
classification_N1_test = classification_N1[:, test_idx]

classification_N1_SSS_train = classification_N1_SSS[:, training_idx]
classification_N1_SSS_test = classification_N1_SSS[:, test_idx]


# COMMENT
# QUESTION: For which of the two databases are we training the DT and NN? 
#           Why don't you try to train a DT and an NN for the other database too? 
print("-------------------------------------------")
print("Train a decision tree and compute accuracy")

# simple decision tree training
# from webpage https://scikit-learn.org/stable/modules/tree.html

# As with other classifiers, DecisionTreeClassifier takes as input two arrays: an array X, sparse or dense, of size [n_samples, n_features] holding the training samples, and an array Y of integer values, size [n_samples], holding the class labels for the training samples:
# X = [[0, 0], [1, 1]]
# Y = [0, 1]
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, Y)
# tree.plot_tree(clf)

#COMMENT - define a decision tree classifier model, and train it on training data (the less-tight criterion one)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(numpy.transpose(generator_dispatch_train), classification_N1_train[0, :])
# plot decision tree
tree.plot_tree(clf)


# COMMENT - run the trained model on test (and training?) data
classification_N1_test_pred = clf.predict(numpy.transpose(generator_dispatch_test))
classification_N1_train_pred = clf.predict(numpy.transpose(generator_dispatch_train))

# COMMENT - calculate the accuracy performance metric of the model on the training dataset
acc_train = 100 * (1 - numpy.sum(numpy.absolute(classification_N1_train_pred - classification_N1_train[0, :])) / (
    len(classification_N1_train[0, :])))

print(f"Decision tree training accuracy: {acc_train} %")
# 100% accuracy was not unexpected for a decision tree model
# COMMENT - calculate the accuracy performance metric of the model, now on yet unseen test data
acc_test = 100 * (1 - numpy.sum(numpy.absolute(classification_N1_test_pred - classification_N1_test[0, :])) / (
    len(classification_N1_train[0, :])))
print(f"Decision tree test accuracy {round(acc_test, 5)} %")
# Quite 'accurate', but as discussed, a different metric should likely be used in this case
# 

# COMMENT 
print("-------------------------------------------")
print("Train a neural network and compute accuracy")

# COMMENT - NN model with 3 hidden layers, 20 nodes each, relu activation function, and for output layer spit out probabilities y_1 and y_2 of whether the system is safe or unsafe
NN = keras.Sequential([
    layers.Dense(20, activation='relu', input_shape=[5]),
    layers.Dense(20, activation='relu'),
    layers.Dense(20, activation='relu'),
    layers.Dense(2, activation='softmax')
])

#COMMENT - implement Adam algorithm for backpropagation, a cheap gradient descent method
optimizer = tf.keras.optimizers.Adam(0.001)
# there are a lot of options for the optimizer
# tf.keras.optimizers.RMSprop(0.001)

# COMMENT - compile the model with the defined NN, using the binary cross-entropy function, and the above-defined optimiser
NN.compile(loss='binary_crossentropy',
           optimizer=optimizer)

history = NN.fit(numpy.transpose(generator_dispatch_train), numpy.transpose(classification_N1_train),
                 epochs=20, verbose=0,
                 validation_data=(numpy.transpose(generator_dispatch_test), numpy.transpose(classification_N1_test)),
                 batch_size=500)

# COMMENT - see what the model predicts on the training (sanity check) and test datasets
classification_N1_test_pred = NN.predict(numpy.transpose(generator_dispatch_test))
classification_N1_train_pred = NN.predict(numpy.transpose(generator_dispatch_train))

# COMMENT - NN training model accuracy not expected to be 100% like with a decision tree
acc_train = 100 * (1 - numpy.sum(numpy.absolute(classification_N1_train_pred[:, 0] - classification_N1_train[0, :])) / (
    len(classification_N1_train[0, :])))

print("Neural network training accuracy", acc_train)
# COMMENT - the NN ends up being more accurate on the test set rather than the data set, explainable by data set sizes and NN behaviour
acc_test = 100 * (1 - numpy.sum(numpy.absolute(classification_N1_test_pred[:, 0] - classification_N1_test[0, :])) / (
    len(classification_N1_train[0, :])))
print("Neural network test accuracy", acc_test)
