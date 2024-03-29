import numpy as np
import matplotlib.pyplot as plt


def combined_activation(x):
    # Leaky ReLU for negative values
    negative_values = np.where(x < 0, 0.01 * x, x)

    # ReLU6 and tanh for positive values with log input
    positive_values = np.where(x > 0, np.minimum(np.maximum(np.tanh(np.log(x + 1)), 0), 6), negative_values)

    return positive_values


# Generate input data
x = np.linspace(-5, 5, 100)

# Compute combined activation function's output
y = combined_activation(x)

# Plot combined activation function
plt.plot(x, y, label='Combined Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Combined Activation Function')
plt.grid(True)
plt.legend()
plt.show()
