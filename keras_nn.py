import sys 
import random
import keras
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn import preprocessing
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import keras.backend as K
#import main 

#import all_Y labels, test_idx, total_X, X_counts  

def my_loss_function(X, X_recon):
	
	loss = K.sum(X - X_recon)

	#Get Y and Ypred
	Y = all_Y[test_idx]
	X_test = total_X[marker: marker+X_counts[i]]
	Ypred = Y #?? 



def classify(X, Y, X_test, input_dim, label_dim):

	classifier = keras.Sequential()
	if label_dim == 2: 
		output_layer_activF = 'sigmoid'
	else:
		output_layer_activF = 'softmax'

	#Hidden layer 
	classifier.add(Dense(input_dim*2, activation='relu', 
								kernel_initializer='random_normal', input_dim=input_dim))
	
	classifier.add(Dense(input_dim*4, activation='relu', 
								kernel_initializer='random_normal', input_dim=input_dim*2))
	#Output layer 
	classifier.add(Dense(label_dim, activation=output_layer_activF, 
								kernel_initializer='random_normal'))
	#Compile model 
	classifier.compile(optimizer ='adam',loss='binary_crossentropy', 
								metrics =['accuracy'])
	print("NN training... ")
	classifier.fit(X, Y, batch_size=5, epochs=200, verbose=0)
	# evaluate the keras model
	_, accuracy = classifier.evaluate(X, Y)
	print('Keras Classifier Accuracy: %.2f' % (accuracy*100))
	# make class predictions with the model
	predictions = classifier.predict_classes(X_test)

	return predictions


def autoencoder(X, X_test, encoding_dim): 

	"""
	Parameters: X: training data, X_test: testing data, encoding_dim: dimension of most hidden layer 

	Return hidden layer representions of training and testing data. 
	"""

	# this is our input placeholder
	input_X = Input(shape=(36,))
	encoded = Dense(24, activation='relu')(input_X)
	encoded = Dense(encoding_dim, activation='relu')(encoded)

	decoded = Dense(24, activation='relu')(encoded)
	decoded = Dense(36, activation='linear')(decoded)

	#encode model
	autoencoder = Model(input_X, decoded)
	
	#separate encoder model
	encoder = Model(input_X, encoded)
	# create a placeholder for an encoded (2-dimensional) input
	#encoded_input = Input(shape=(encoding_dim,))
	# retrieve layers of the autoencoder model
	#decoder_layer_1 = autoencoder.layers[1]
	#decoder_layer_2 = autoencoder.layers[0]
	# create the decoder model
	#decoder = Model(encoded_input, decoder_layer_2(decoder_layer_1(encoded_input)))
	X_train, X_valid  = train_test_split(X,
										test_size=0.2,
										random_state=13)
	#Normalize 
	X_valid = preprocessing.normalize(X_valid)
	X_train = preprocessing.normalize(X_train)
	X_test = preprocessing.normalize(X_test)
	X = preprocessing.normalize(X)

	#loss = keras.losses.categorical_crossentropy // loss='binary_crossentropy'
	#optimizer=keras.optimizers.Adam() // optimizer='adadelta'
	autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
	#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
	autoencoder_train = autoencoder.fit(X_train, X_train,
											epochs=10,
											batch_size=125,
											shuffle=True,
											validation_data=(X_valid, X_valid),
											verbose=2)

	#model_json = autoencoder.to_json()
	#with open("model_clf_" + str(count) + ".json", "w") as json_file:
	#json_file.write(model_json)
	# serialize weights to HDF5
	#autoencoder.save_weights("autoencoder_clf_" + str(count) + ".h5")
	#print("Saved model to disk")
	print("autoencoder training done. ")

	hidden_train = encoder.predict(X)
	hidden_test = encoder.predict(X_test)

	print("autoencoder predicting done. ")

	return hidden_train, hidden_test








