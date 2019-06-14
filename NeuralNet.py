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
		self.biases = np.random.rand(nextNuerons, 1)
		self.weights = np.random.rand(nextNuerons, actualization.size)
		self.actuals = actualization
		self.nextActuals = np.empty([nextNuerons, 1])


	def getNextActuals(self):
		self.nextActuals = np.add(np.matmul(self.weights, self.actuals), self.biases);
		for x in np.nditer(self.nextActuals.size):
			x = sigmoid(x)
		return self.nextActuals

	def setActuals(self, newActuals):
		self.actuals = newActuals;

	def setWeightsAndBias(gradientWeight, gradientBias):
		weightsVector = getWeightVector()

		newWeights = np.add(weightsVector, gradientWeight)
		newBiases = np.add(self.biases, gradientBias) 

		self.biases = newBiases
		self.weights = np.reshape(newWeights, (weightDimension[0], weightDimension[1]))

	def getWeightVector(self):
		weightsVector = np.reshape(self.weights, (self.weights.size, 1))
		return weightsVector

	def getBiases(self):
		return self.biases




class NueralNetwork:
	def __init__(self):
		self.InputLayer = None
		self.HiddenLayers = list()
		self.OutputLayer = None
		self.Layers = list()

		self.layerCount = 0
		self.dataCount = 0

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

	def predict(self, dataEntry):
		for layer in self.Layers:
			if layer is self.InputLayer:
				layer.setActuals = dataEntry
				previousLayer = layer
			else:
				layer.setActuals = previousLayer.getNextActuals()
				previousLayer = layer

		self.makePrediction();

	def makePrediction(self):
		finalPrediction = 0;
		prediction = 0;
		for x in np.nditer(self.OutputLayer.actuals):
			if x > prediction:
				finalPrediction = prediction
			prediction += 1

		return finalPrediction, self.OutputLayer.actuals

	def fit(self, data, answers):
		previousLayer = None
		self.InputLayer.setActuals(data)

		for dataEntry in data.T:
			self.predict(dataEntry);
			self.backPropogate(self.makePrediction(), answers);
		self.dataCount = 0

	def costCalculation(self, results, correctResult):
		error = 0.0
		predictedResult = 0
		resultNumber = 0
		resultIndex = 0
		indexIterator = 0

		for check in np.nditer(results[1]):
			print("check: ", check)
			if check > predictedResult:
				predictedResult = check
				resultIndex = indexIterator

			indexIterator += 1;

		print("resultIndex: ", resultIndex)
		print("Result: ", predictedResult)
		print("Correct: ", correctResult[self.dataCount])


		for result in np.nditer(results[1]):

			if indexIterator != correctResult[self.dataCount]:
				error = error + (result - 0.0)**2

			else:
				error = error + (result - 1.0)**2
			resultNumber += 1;

		resultNumber += 1
		error = error / resultNumber
		print("Error: ", error)
		self.dataCount += 1

		return error;



	def backPropogate(self, prediction, correctResult):
		for layer in reversed(self.Layers):
			if(layer.weights.size == 0):
				continue
			errorWeight = np.full((layer.weights.size, 1), self.costCalculation(prediction, correctResult))
			errorBias = np.full((layer.biases.size, 1), self.costCalculation(prediction, correctResult))
			print(tuple(map(tuple, layer.getWeightVector())))
			gradientWeight = np.gradient(errorWeight, tuple(map(tuple, layer.getWeightVector())))
			gradientBias = np.gradient(errorBias, layer.getBiases())
			layer.setWeightsAndBias(gradientWeight, gradientBias)


def intializeNueralNetwork(data, outputs, hiddenLayers, hiddenNuerons):
	NN = NueralNetwork();
	NN.setInputLayer(Layer(data, data.size, hiddenNuerons))
	for x in range(hiddenLayers):
		NN.setHiddenLayers(Layer(np.random.rand(hiddenNuerons, 1), hiddenNuerons, hiddenNuerons))
	NN.setOutputLayer(Layer(np.random.rand(outputs, 1), 0, 0))	
	return NN;



myNueralNetwork = intializeNueralNetwork(np.random.rand(28, 1), 10, 2, 20)

myNueralNetwork.fit(x_train, y_train)
print(myNueralNetwork.getInputLayer())
print(myNueralNetwork.getHiddenLayers(0))
print(myNueralNetwork.getHiddenLayers(1))
print(myNueralNetwork.getOutputLayer())
