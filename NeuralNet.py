import numpy as np
import math
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class Layer:
	def __init__(self, actualization, nuerons, nextNuerons):
		self.weightDimension = [nuerons, nextNuerons]
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

	def setWeightsAndBias(gradientWeight, gradientBias):
		weightsVector = getWeightVector()

		newWeights = np.add(weightsVector, gradientWeight)
		newBiases = np.add(self.biases, gradientBias) 

		self.biases = newBiases
		self.weights = np.reshape(newWeights, (weightDimension[0], weightDimension[1]))

	def getWeightVector():
		weightsVector = np.reshape(self.weights, (self.weights.size, 1))
		return weightsVector

	def getBiases():
		return self.biases




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

	def fit(self, data, answers):
		previousLayer = None

		for dataEntry in data.T:
			predict(dataEntry);
			backPropogate(makePrediction(), answers);


	def costCalculation(results, correctResult):
		error = 0.0;
		for result in np.nditer(results.actuals):
			if resultNumber != correctResult:
				error = error + (result - 0.0)**2

			else:
				error = error + (result - 1.0)**2
			resultNumber += 1;

		resultNumber += 1;
		error = error / resultNumber

		return error;



	def backPropogate(prediction, correctResult):
		for layer in reversed(self.Layers):
			errorWeight = np.full((layer.weights.size, 1), costCalculation(prediction, correctResult))
			errorBias = np.full((layer.biases.size, 1), costCalculation(prediction, correctResult))
			gradientWeight = np.gradient(errorWeight, layer.getWeightVector())
			gradientBias = np.gradient(errorBias, layer.getBiases())
			layer.setWeightsAndBias(gradientWeight, gradientBias)


def intializeNueralNetwork(data, inputs, outputs, hiddenLayers, hiddenNuerons):
	NN = NueralNetwork();
	NN.setInputLayer(Layer(data, inputs, hiddenNuerons))
	for x in range(hiddenLayers):
		NN.setHiddenLayers(Layer(np.empty([hiddenNuerons, 1]), hiddenNuerons, hiddenNuerons))
	NN.setOutputLayer(Layer(np.empty([outputs, 1]), 0, 0))	
	return NN;



myNueralNetwork = intializeNueralNetwork(np.empty([28, 1]), 20, 5, 2, 7)
myNueralNetwork.fit(x_train, y_train)
print(myNueralNetwork.getInputLayer())
print(myNueralNetwork.getHiddenLayers(0))
print(myNueralNetwork.getHiddenLayers(1))
print(myNueralNetwork.getOutputLayer())
