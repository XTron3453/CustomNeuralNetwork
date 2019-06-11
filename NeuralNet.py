import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class Layer:
	def __init__(self, actualization, nuerons, nextNuerons):
		self.biases = np.random.rand(nuerons, 1)
		self.weights = np.random.rand(nuerons, nextNuerons)
		self.actuals = actualization
		self.nextActuals = np.empty([nextNuerons, 1])


	def getNextActuals():
		self.nextActuals = np.add(np.matmultiply(self.weights, self.actuals), self.biases);
		for x in np.nditer(nextNuerons):
			x = sigmoid(x)
		return self.nextActuals

	def setActuals(newActuals):
		self.actuals = newActuals;

	#def setWeights():

class NueralNetwork:
	def __init__(self):
		self.InputLayer = None
		self.HiddenLayers = list()
		self.OutputLayer = None
		self.Layers = list()
		self.layerCount = 0


	def setInputLayer(self, inputLayer):
		self.InputLayer = inputLayer
		Layers.append(inputLayer)
		self.layerCount += 1

	def setHiddenLayers(self, NewLayer):
		self.HiddenLayers.append(NewLayer)
		self.Layers.append(NewLayer)
		self.layerCount += 1

	def setOutputLayer(self, outputLayer):
		self.OutputLayer = outputLayer
		self.Layers.append(outputLayer)
		self.layerCount += 1

	def getInputLayer(self):
		return self.InputLayer

	def getHiddenLayers(self, index):
		return self.HiddenLayers[index]
	
	def getOutputLayer(self):
		return self.OutputLayer

	def predict(dataEntry):
		for layer in Layers:
			if layer is self.InputLayer:
				layer.setActuals = dataEntry
				previousLayer = layer
			else:
				layer.setActuals = previousLayer.getNextActuals()
				previousLayer = layer

		makePrediction();

	def makePrediction(self):
		finalPrediction = 0;
		prediction = 0;
		actualization = 0;
		for x in np.nditer(self.OutputLayer):
			if x > prediction:
				finalPrediction = prediction
			prediction += 1

		return finalPrediction, actualization

	def fit(self, train, test):
		previousLayer = None

		for dataEntry in train.T:
			predict();
			backPropogate(makePrediction());


	def backPropogate(prediction):
		print('TBI')



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
