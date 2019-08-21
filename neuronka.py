import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

training_input = np.array([[0, 0, 1],
						   [1, 1, 1],
						   [1, 0, 1],
						   [0, 1, 0]])

training_outputs = np.array ([[0, 1, 1, 0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

print("Случайные инициализирующие веса:")
print(synaptic_weights)

# Training of my girl

for i in range(20000):
	input_layers = training_input
	outputs = sigmoid(np.dot(input_layers, synaptic_weights))

	err = training_outputs - outputs
	adjusment = np.dot(input_layers.T, err * (outputs *  (1 - outputs)))
	synaptic_weights += adjusment

# print("Результат:")
# print(outputs)

new_inputs = np.array([0, 1, 1])
output = sigmoid( np.dot(new_inputs, synaptic_weights))

print("Input: ")
print(new_inputs)

print("Результат:")
print(output)