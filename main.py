import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import time

def create_model():
    model = Sequential()
    model.add(Dense(4, input_dim=2, activation='sigmoid'))
    model.add(Dense(3, activation='sigmoid'))
    return model

def visualize_network(model, input_data):
    layers = len(model.layers)
    weights = [layer.get_weights() for layer in model.layers]

    plt.figure(figsize=(10, 5))

    for i in range(layers):
        layer_weights, layer_biases = weights[i]
        neurons = layer_weights.shape[1]

        for j in range(neurons):
            neuron_weights = layer_weights[:, j]
            neuron_bias = layer_biases[j]

            if i == 0:  # input layer
                neuron_input = input_data
            else:
                neuron_input = np.dot(input_data, weights[i - 1][0]) + weights[i - 1][1]

            neuron_output = neuron_input * neuron_weights + neuron_bias
            neuron_activation = 1 / (1 + np.exp(-neuron_output))

            color = 'red' if np.any(neuron_activation > 0.5) else 'blue'

            # Calculate the average of neuron_activation values and use it as alpha
            alpha = np.mean(neuron_activation)

            plt.scatter(i + 1, j, s=100, c=color, alpha=alpha)

            # Format each activation value in the array
            activation_text = ', '.join([f"{value:.2f}" for value in neuron_activation])
            plt.text(i + 1, j, activation_text, fontsize=12, ha='center', va='center', color='white')

            if i > 0:
                prev_layer_neurons = weights[i - 1][0].shape[1]

                for k in range(prev_layer_neurons):
                    plt.plot([i, i + 1], [k, j], c=color, alpha=alpha)

    plt.xlim(0, layers + 1)
    plt.yticks([])
    plt.xlabel("Layer")
    plt.ylabel("Neuron")
    plt.title("Neural Network Visualization")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

model = create_model()
input_data = np.array([1, 1])
visualize_network(model, input_data)
