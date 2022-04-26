import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dbn.tensorflow import SupervisedDBNClassification
import pandas as pd 
import matplotlib.pyplot as plt

# Loading the dataset: 
attacks_data = pd.read_csv(r'C:\Users\DELL\Desktop\deep-belief-network-master\attacks_data.csv', header = None, engine = 'python', encoding = 'latin-1')
print(attacks_data.head())
a=np.shape(attacks_data)
print(a)
print(attacks_data.head())
Y = attacks_data[9]
X = attacks_data.drop([9], axis=1)

#digits = load_digits()
#X, Y = digits.data, digits.target

# Data scaling
#X = (X / 16).astype(np.float32)
X = X.astype(np.float32)
X=X.to_numpy()
# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)

# Test
Y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))
