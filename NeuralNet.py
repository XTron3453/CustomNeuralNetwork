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

	def setWeights(gradient):
		self.weights = np.add(self.weights, gradient)


	def getWeightsAndBiasVector():
		weightCounter = 0
		biasCounter = 0
		numberOfItems = 0;
		size = self.weights.size + self.biases.size
		weightsAndBiases = np.empty([size ,1])
		for item in np.nditer(weightsAndBiases):
			if numberOfItems % 2 == 0:
				item = self.weights.item(weightCounter)
				weightCounter += 1
			
			else:
				item = self.biases.item(biasCounter)
				biasCounter += 1

			numberOfItems += 1

		return weightsAndBiases



class NueralNetwork:
	def __init__(self):
		self.InputLayer = None
		self.HiddenLayers = list()
		self.OutputLayer = None
		self.Layers = list()
		self.layerCount = 0


	def setInputLayer(self, inputLayer):
		self.InputLayer = inputLayer
		self.Layers.append(inputLayer)
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
			predict(dataEntry);
			backPropogate(makePrediction(), test);


	def errorCalculation(results, correctResult):
		error = 0.0;
		resultNumber = 0;
		for result in np.nditer(results.actuals):
			if resultNumber != correctResult:
				error = error + (result - 0.0)**2

			else:
				error = error + (result - 1.0)**2
			resultNumber += 1;

		resultNumber += 1;
		error = error / resultNumber;
		return error;



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
