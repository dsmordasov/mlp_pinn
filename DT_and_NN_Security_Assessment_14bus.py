# all these modules are necessary
import numpy
from sklearn import tree
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# insert the directory where the data is stored
data_directory = ""

# COMMENT
print("Loading generator set-points -- features for machine learning")
# ATTENTION: The MAT files need to be converted to CSV files to be imported (see MATLAB fct for that)
generator_dispatch = numpy.loadtxt(open(data_directory + "generator_dispatch.csv", "rb"), delimiter=",", skiprows=0)
print(generator_dispatch)
print(generator_dispatch.shape)
# normalize features for better performance

# COMMENT
print("Loading  security classification N-1 security")
classification_N1 = numpy.loadtxt(open(data_directory + "classification_N1.csv", "rb"), delimiter=",", skiprows=0)
# COMMENT
classification_N1 = numpy.row_stack(([classification_N1], [numpy.ones(classification_N1.shape) - classification_N1]))
print(classification_N1)
print("The number of secure samples:", numpy.sum(classification_N1))


# COMMENT
print("security classification N-1 security and small-signal stability")
classification_N1_SSS = numpy.loadtxt(open(data_directory + "classification_N1_SSS.csv", "rb"), delimiter=",",
                                      skiprows=0)
# COMMENT
classification_N1_SSS = numpy.row_stack(
    ([classification_N1_SSS], [numpy.ones(classification_N1_SSS.shape) - classification_N1_SSS]))
print(classification_N1_SSS)
print(numpy.sum(classification_N1))
print("The number of secure samples:", numpy.sum(classification_N1_SSS))

# COMMENT
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
print("train a decision tree and compute accuracy")

# simple decision tree training
# from webpage https://scikit-learn.org/stable/modules/tree.html

# As with other classifiers, DecisionTreeClassifier takes as input two arrays: an array X, sparse or dense, of size [n_samples, n_features] holding the training samples, and an array Y of integer values, size [n_samples], holding the class labels for the training samples:
# X = [[0, 0], [1, 1]]
# Y = [0, 1]
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, Y)
# tree.plot_tree(clf)

#COMMENT
clf = tree.DecisionTreeClassifier()
clf = clf.fit(numpy.transpose(generator_dispatch_train), classification_N1_train[0, :])
# plot decision tree
# tree.plot_tree(clf)


# COMMENT
classification_N1_test_pred = clf.predict(numpy.transpose(generator_dispatch_test))
classification_N1_train_pred = clf.predict(numpy.transpose(generator_dispatch_train))

# COMMENT
acc_train = 100 * (1 - numpy.sum(numpy.absolute(classification_N1_train_pred - classification_N1_train[0, :])) / (
    len(classification_N1_train[0, :])))

print("Decision tree training accuracy", acc_train)
# COMMENT
acc_test = 100 * (1 - numpy.sum(numpy.absolute(classification_N1_test_pred - classification_N1_test[0, :])) / (
    len(classification_N1_train[0, :])))
print("Decision tree test accuracy", acc_test)

# COMMENT
print("-------------------------------------------")
print("train a neural network and compute accuracy")

# COMMENT
NN = keras.Sequential([
    layers.Dense(20, activation='relu', input_shape=[5]),
    layers.Dense(20, activation='relu'),
    layers.Dense(20, activation='relu'),
    layers.Dense(2, activation='softmax')
])

#COMMENT
optimizer = tf.keras.optimizers.Adam(0.001)
# there are a lot of options for the optimizer
# tf.keras.optimizers.RMSprop(0.001)

# COMMENT
NN.compile(loss='binary_crossentropy',
           optimizer=optimizer)

history = NN.fit(numpy.transpose(generator_dispatch_train), numpy.transpose(classification_N1_train),
                 epochs=20, verbose=0,
                 validation_data=(numpy.transpose(generator_dispatch_test), numpy.transpose(classification_N1_test)),
                 batch_size=500)

# COMMENT
classification_N1_test_pred = NN.predict(numpy.transpose(generator_dispatch_test))
classification_N1_train_pred = NN.predict(numpy.transpose(generator_dispatch_train))

# COMMENT
acc_train = 100 * (1 - numpy.sum(numpy.absolute(classification_N1_train_pred[:, 0] - classification_N1_train[0, :])) / (
    len(classification_N1_train[0, :])))

print("Neural network training accuracy", acc_train)
# COMMENT
acc_test = 100 * (1 - numpy.sum(numpy.absolute(classification_N1_test_pred[:, 0] - classification_N1_test[0, :])) / (
    len(classification_N1_train[0, :])))
print("Neural network test accuracy", acc_test)
