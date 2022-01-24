from Preprocesare import Procesare
from NNRating import NNRating
import json
from keras.models import load_model
import numpy as np

def predict(nnModel, procesare, max_length, sentence):
	prediction = nnModel.predict(np.array([procesare.processSentence(sentence, max_length)]))[0]
	ERROR = 0.3
	results = [[i,r] for i,r in enumerate(prediction) if r>ERROR]
	results.sort(key=lambda x: x[1], reverse=True)
	return procesare.getRatings()[results[0][0]]

def predictSentences(nnModel, procesare, max_length):
	f = open('test.json')
	fw = open('testWrite.json', 'w')
	dataset = json.load(f)
	for i in range(len(dataset)):
		rating = predict(nnModel,procesare,max_length,dataset[i]['text'])
		dataset[i]['rating'] = rating

	f.close()
	json.dump(dataset, fw)
	fw.close()	

def main():
	procesare = Procesare()
	procesare.readIntents()
	procesare.trainWordEmbeddings()
	#training_data = []
	(training_data_X, training_data_Y, vocab_size, max_length) = procesare.training_set()
	(embedding_matrix, num_words) = procesare.readEmbeddings(max_length)

	#neuralNetwork = NNRating(training_data_X, embedding_matrix, vocab_size, max_length, training_data_Y)
	#neuralNetwork.NNEmbedding()
	#neuralNetwork.fitModelEmbedding()

	nnModel = load_model('Rating_model.h5')
	#predict(nnModel,procesare, max_length, "Produs Ok Se simte bine in mana Se pot descarca drivere de la producator si se pot programa butoanele")
	predictSentences(nnModel,procesare, max_length)
if __name__ == '__main__':
	main()