import numpy as np
from train_scripts import *
from plot_cost import *
import matplotlib.pyplot as plt

figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]
              ])



y = np.array([0, 1, 1, 0])

y2 = np.array([0, 1, 1, 1]) # target outputs for OR

n_hidden = 2
iterations = 1000
learning_rate = 0.01
learning_rate2 = 0.5

# Train the neural network on the XOR problem
# For now, we will not use the 3rd and 4th outputs of the function, hence we use "_" on the returned outputs
errors, nn, _, _ = train(X, y, n_hidden, iterations, learning_rate)
errors2, nn2, _, _ = train(X, y2, n_hidden, iterations, learning_rate2)
# Test the neural network on the XOR problem
test_xor(X, y, nn)
test_xor(X, y2, nn)

# Plot the cost for all iterations
fig, ax1 = plt.subplots()
plot_cost(errors, ax1)
fig2, ax2 = plt.subplots()
plot_cost(errors2, ax2)
plot_filename = os.path.join(os.getcwd(), 'figures', 'XOR_cost.png')
plot_filename2 = os.path.join(os.getcwd(), 'figures', 'OR_cost.png')
plt.savefig(plot_filename)
plt.savefig(plot_filename2)
min_cost = np.min(errors)
argmin_cost = np.argmin(errors)
print('Minimum cost: {:.5f}, on iteration #{}'.format(min_cost, argmin_cost+1))
min_cost = np.min(errors2)
argmin_cost = np.argmin(errors2)
print('Minimum cost: {:.5f}, on iteration #{}'.format(min_cost, argmin_cost+1))

# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()
