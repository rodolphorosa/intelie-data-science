import tensorflow as tf 

from keras.models import Sequential 
from keras.layers import Dense 
from keras import backend as K, optimizers 
from sklearn.model_selection import KFold 
from scipy import sparse 

import numpy as np 


n_users = 6040
n_movies = 3952

R = np.zeros((n_users, n_movies))

with open("ratings.dat") as file:
    for line in file.readlines():
        user_id, movie_id, rating, timestamp = line.split("::")
        R[int(user_id) - 1, int(movie_id) - 1] = rating

R = sparse.csr_matrix(R)


def mae(y_true, y_pred):
    return K.mean(K.abs(y_true[y_true > 0] - y_pred[y_true > 0]))


model = Sequential()
model.add(Dense(128, input_dim=3952, activation="relu", use_bias=True))
model.add(Dense(3952, activation="linear"))
sgd = optimizers.SGD(learning_rate=0.001, momentum=0.9)
model.compile(loss=mae, optimizer=sgd)


kf = KFold(n_splits=5, random_state=None, shuffle=False)

evaluations = []

for train, test in kf.split(R):
    errors = []
    
    X_train = R[train]
    X_test = R[test]
    
    model.fit(X_train, X_train, epochs=100, verbose=2)
    
    evaluations.append(model.evaluate(X_test, X_test))

np.mean(evaluations)