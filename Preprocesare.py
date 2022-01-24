import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import random
import string
from nltk.corpus import stopwords
import json
import simplemma
from nltk.tokenize import word_tokenize
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os


class Procesare:
	def __init__(self):
		self.lemmatizer = simplemma.load_data('ro')
		self.remove_punctuation = [punct for punct in string.punctuation]
		self.f = open('train.json')
		self.dataset = json.load(self.f)
		self.ratings = [1,2,3,4,5]
		self.sentences = []
		self.review_lines = list()
		self.ratingsDataset = []
		self.tokenizer_obj = Tokenizer()

	def getRatings(self):
		return self.ratings;

	def LemTokens(self):
		return [simplemma.lemmatize(w.lower(), self.lemmatizer) for w in self.words if w not in self.remove_punctuation]

	def filterStopW(self, words):
		stops = set(stopwords.words('romanian'))
		tokens_wo_stopwords = [t for t in words if t not in stops]
		return tokens_wo_stopwords


	def readIntents(self):
		for datasetItem in self.dataset:
			w = self.filterStopW(nltk.word_tokenize(datasetItem['text']))
			w = [simplemma.lemmatize(word.lower(), self.lemmatizer) for word in w]
			self.ratingsDataset.append(datasetItem['rating'])
			self.sentences.append(datasetItem['text'])

	def predictTheTestFile(self, max_length):
		f = open('test.json')
		dataset = json.load(f)
		sentences = []
		processedSentences = []
		for datasetItem in dataset:
			w = self.filterStopW(nltk.word_tokenize(datasetItem['text']))
			w = [simplemma.lemmatize(word.lower(), self.lemmatizer) for word in w]
			sentences.append(datasetItem['text'])
		for sentence in sentences:
			processedSentences.append(self.processSentence(sentence, max_length))
		f.close()
		return processedSentences


	def trainWordEmbeddings(self):
		for review in self.sentences:
			tokens = word_tokenize(review)
			tokens = [w.lower() for w in tokens]
			words = [simplemma.lemmatize(word, self.lemmatizer) for word in tokens]
			words_without_punctation = [t for t in words if t not in self.remove_punctuation]
			words = self.filterStopW(words_without_punctation)
			self.review_lines.append(words)
		model = gensim.models.Word2Vec(sentences = self.review_lines, vector_size=100, window = 5, workers = 4, min_count = 1)
		fileName = 'word_embeddings.txt'
		model.wv.save_word2vec_format(fileName, binary = False)

	def readEmbeddings(self, max_length):
		embeddings_index = {}
		f = open(os.path.join('', 'word_embeddings.txt'), encoding = "utf-8")
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:])
			embeddings_index[word] = coefs
		f.close()
		self.tokenizer_obj.fit_on_texts(self.review_lines)
		sequences = self.tokenizer_obj.texts_to_sequences(self.review_lines)
		word_index = self.tokenizer_obj.word_index
		review_pad = pad_sequences(sequences, maxlen=max_length)
		num_words = len(word_index) + 1
		embedding_matrix = np.zeros((num_words, 100))
		for word, i in word_index.items():
			if i > num_words:
				continue
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector
		return embedding_matrix, num_words

	def training_set(self):
		self.tokenizer_obj.fit_on_texts(self.review_lines)
		training_data = self.tokenizer_obj.texts_to_sequences(self.review_lines)
		max_length = 0
		for review in self.review_lines:
			maxlen = len(review)
			if max_length < maxlen:
				max_length = maxlen
		vocab_size = len(self.tokenizer_obj.word_index) + 1
		training_data_X = pad_sequences(training_data, maxlen = max_length, padding = 'post')
		training_data_Y = []
		for ratingItem in self.ratingsDataset:
			output_row = [0] * len(self.ratings)
			output_row[self.ratings.index(ratingItem)] = 1
			training_data_Y.append(output_row)
		return training_data_X, training_data_Y, vocab_size, max_length


	def processSentence(self, sentence, max_length):
		tokens = word_tokenize(sentence)
		tokens = [w.lower() for w in tokens]
		words = [simplemma.lemmatize(word, self.lemmatizer) for word in tokens]
		stripped = [t for t in words if t not in self.remove_punctuation]
		words = self.filterStopW(stripped)
		sequence = self.tokenizer_obj.texts_to_sequences([words])
		sequence = pad_sequences(sequence, maxlen = max_length, padding = 'post')
		return (sequence[0])

