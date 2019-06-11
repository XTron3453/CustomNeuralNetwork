import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class Layer:
	def __init__(self, actualization, nuerons, nextNuerons):
		self.biases = np.random.rand(nuerons, 1)
		self.weights = np.random.rand(nuerons, nextNuerons)
		self.actuals = actualization
		self.newActuals = np.empty([nextNuerons, 1])


	def getNewActuals():
		self.newActuals = np.add(np.matmultiply(self.weights, self.actuals), self.biases);
		for x in np.nditer(nextNuerons):
			x = sigmoid(x)
		return self.newActuals

	#def setWeights():

class NueralNetwork:
	def __init__(self):
		self.InputLayer = None
		self.HiddenLayers = list();
		self.OutputLayer = None

	def setInputLayer(self, inputs):
		self.InputLayer = inputs

	def setHiddenLayers(self, layer):
		self.HiddenLayers.append(layer)
	
	def setOutputLayer(self, outputs):
		self.OutputLayer = outputs

	def getInputLayer(self):
		return self.InputLayer

	def getHiddenLayers(self, index):
		return self.HiddenLayers[index].weights.size
	
	def getOutputLayer(self):
		return self.OutputLayer

	def makePrediction(self):
		realPrediction;
		prediction = 0;
		actualization = 0;
		for x in np.nditer(self.OutputLayer):
			if x > prediction:
				realPrediction = prediction;
			++prediction

		return realPrediction, actualization;

	#def backPropogate():



def intializeNueralNetwork(data, inputs, outputs, hiddenLayers, hiddenNuerons):
	NN = NueralNetwork();
	NN.setInputLayer(Layer(data, inputs, hiddenNuerons))
	for x in range(hiddenLayers):
		NN.setHiddenLayers(Layer(np.empty([hiddenNuerons, 1]), hiddenNuerons, hiddenNuerons))
	NN.setOutputLayer(Layer(np.empty([outputs, 1]), 0, 0))	
	return NN;



myNueralNetwork = intializeNueralNetwork(np.empty([20, 1]), 20, 5, 2, 7)
print(myNueralNetwork.getInputLayer())
print(myNueralNetwork.getHiddenLayers(0))
print(myNueralNetwork.getHiddenLayers(1))
print(myNueralNetwork.getOutputLayer())
