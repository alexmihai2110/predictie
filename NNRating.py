from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, GRU, LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.initializers import Constant

class NNRating:
	def __init__(self, training_data_X, embedding_matrix, num_words, max_length, training_data_Y):
		self.training_data_x = training_data_X
		self.training_data_y = training_data_Y
		self.model = Sequential()
		self.num_words = num_words
		self.embedding_matrix = embedding_matrix
		self.max_length = max_length


	def NNEmbedding(self):
		self.model.add(Embedding(self.num_words, 100, weights = [self.embedding_matrix], input_length = self.max_length, trainable = False))
		self.model.add(Bidirectional(LSTM(128)))
		self.model.add(Dense(32, activation='relu'))
		self.model.add(Dense(5, activation='softmax'))

		self.model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

	def fitModelEmbedding(self):
		hist = self.model.fit(np.array(self.training_data_x), np.array(self.training_data_y), batch_size=128, epochs=25, verbose = 1)
		self.model.save('Rating_model.h5', hist)