import numpy as np 
import time

# AND
#    1 1 = 1 
#    1 0 = 0
#    0 1 = 0
#    0 0 = 0
# OR
#    1 1 = 1
#    1 0 = 1
#    0 1 = 1
#    0 0 = 0 

data_train = np.array([[0, 0],
					   [1, 0],
					   [0, 1],
					   [1, 1]])
target_train_and = [0, 0, 0, 1]
target_train_or = [0, 1, 1, 1]
data_test = np.array([[0, 1],
					  [1, 1],
					  [0, 0],
					  [1, 0]])

class NeuralNetwork():
	"""docstring for NeuralNetwork"""
	def __init__(self):
		#np.random.seed(25)
		self.weights = np.random.rand(2, 1)
		self.bias = 1
		self.epochs = 50
		self.lr = 0.05

	def activation(self, y):
		if y < 0:
			return 0
		else:
			return 1

	def fit(self, data_train, target_train):

		for epoch in range(0, self.epochs):
			for data, target in zip(data_train, target_train):

				#multiplicacao de matrizes (entradasxpesos) e soma com bias
				out = (data.dot(self.weights) + self.bias)
				#funcao de ativacao do neuronio
				y_pred = self.activation(out)

				#calculo do erro
				erro = target - y_pred
				#calculo da variação do ajuste dos pesos
				delta = erro*data
				#multiplica pela taxa de aprendizagem
				temp = delta*self.lr
				temp = temp.reshape(2,1)
				#atualiza os pesos e o bias
				self.weights = self.weights + temp
				self.bias = self.bias + self.lr*erro


	def predict(self, data_test):

		result = []
		for data in data_test:
			out = data.dot(self.weights) #multiplicacao de matrizes
			v0 = out + self.bias
			y_pred = self.activation(v0)
			result.append(y_pred)

		print(result)

model_and = NeuralNetwork()
model_and.fit(data_train, target_train_and)
model_and.predict(data_test) #resultado deve ser 0 1 0 0 


model_or = NeuralNetwork()
model_or.fit(data_train, target_train_or)
model_or.predict(data_test) #resultado deve ser 1 1 0 1 
