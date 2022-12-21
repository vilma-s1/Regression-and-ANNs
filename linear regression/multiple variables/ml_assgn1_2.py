from load_data_ex2 import *
from normalize_features import *
from gradient_descent import *
from calculate_hypothesis import *
import os

figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# This loads our data
X, y = load_data_ex2()

# Normalize
X_normalized, mean_vec, std_vec = normalize_features(X)


# After normalizing, we append a column of ones to X, as the bias term
column_of_ones = np.ones((X_normalized.shape[0], 1))
# append column to the dimension of columns (i.e., 1)
X_normalized = np.append(column_of_ones, X_normalized, axis=1)

# initialise trainable parameters theta, set learning rate alpha and number of iterations
theta = np.zeros((3))
alpha = 0.001
iterations = 100

# plot predictions for every iteration?
do_plot = True

# call the gradient descent function to obtain the trained parameters theta_final
theta_final = gradient_descent(X_normalized, y, theta, alpha, iterations, do_plot)
print(theta_final)

#########################################
# Write your code here

sample_1 = [1650, 3]
sample_2 = [3000, 4]
normalized_sample_1 = np.zeros(2)
normalized_sample_2 = np.zeros(2)

normalized_sample_1[0] = (sample_1[0]- mean_vec[0, 0])/std_vec[0, 0]
normalized_sample_1[1] = (sample_1[1]- mean_vec[0, 1])/std_vec[0, 1]
normalized_sample_2[0] = (sample_2[0]- mean_vec[0, 0])/std_vec[0, 0]
normalized_sample_2[1] = (sample_2[1]- mean_vec[0, 1])/std_vec[0, 1]

prediction_1 = theta_final[0] + normalized_sample_1[0]*theta_final[1] + normalized_sample_1[1]*theta_final[2]
prediction_2 = theta_final[0] + normalized_sample_2[0]*theta_final[1] + normalized_sample_2[1]*theta_final[2]

print('Prediction 1: {:.5f}'.format(prediction_1))
print('Prediction 2: {:.5f}'.format(prediction_2))

# print(prediction_1)
# print(prediction_2)

# = normalize_features(sample_1)
#normalization_values_2 = normalize_features(sample_1)

#means_1 = normalization_values_1[1]
#means_2 = normalization_values_2[1]
#stds_1 = normalization_values_1[2]
#stds_2 = normalization_values_2[2]


#normalized_sample_1[0] = (sample_1[0]- means_1[0])/stds_1[0]
#normalized_sample_1[1] = (sample_1[1]- means_1[1])/stds_1[1]
#normalized_sample_2[0] = (sample_2[0]- means_2[0])/stds_2[0]
#normalized_sample_2[1] = (sample_2[1]- means_2[1])/stds_2[1]

# Create two new samples: (1650, 3) and (3000, 4)
# Calculate the hypothesis for each sample, using the trained parameters theta_final
# Make sure to apply the same preprocessing that was applied to the training data
# Print the predicted prices for the two samples

########################################/
