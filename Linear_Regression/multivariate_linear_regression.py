"""
Source: https://machinelearningmedium.com/2017/08/23/multivariate-linear-regression/
"""
import numpy as np

print(__doc__)

"""
Dummy data for Multivariate Regression
"""
X_train = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 5]]).T
Y_train = np.array([[1, 2, 4, 3, 5.5, 8, 6, 8.4, 10, 4]]).T


"""
Hypothesis Function
"""
def h_function(x, theta):
	# theta(k, 1) x(k, 1)
	# print(theta.shape, x.T.shape)
	# x = np.array(x).reshape(-1, 1)
	# print(theta.T, x)
	# print("MAT: ", np.matmul(theta, x.T)[0][0])
	# exit(0)
	return np.matmul(theta, x.T)[0][0]

"""
Update features by order of the regression
[x] -> [1, x, x^2, ..., x^order_of_regression ]
"""
def update_features(x, order_of_regression):
	# 
	x_feature = [1.0]
	value = 1.0
	for i in range(order_of_regression):
		value = value * x[0]
		x_feature.append(value)
	return np.atleast_2d(x_feature)

"""
Cost function/ Loss function J(theta)
"""
def loss_function_j(X_train, Y_train, theta, order_of_regression):
	cost = 0
	m = X_train.shape[0]
	for (x, y) in zip(X_train, Y_train):
		x_feature = update_features(x, order_of_regression)
		# print("X_feature: {}".format(x_feature))
		cost +=  (h_function(x_feature, theta) - y) ** 2
		# print("cost: {} regular: {}".format(cost, (1.0 / (2.0 * m) ) * cost))
	# exit(0)
	return (1.0 / (2.0 * m) ) * cost

"""
Partial Derivative w.r.t. theta_i
"""
def derivative_theta_i(X_train, Y_train, theta, order_of_regression, i):
	result = 0
	m = X_train.shape[0]
	for (x, y) in zip(X_train, Y_train):
		# print(x, y)
		x_feature = update_features(x, order_of_regression)
		# print("x_feature: {}".format(x_feature))
		# print("h_function:", h_function(x_feature, theta) - y[0], y)
		# loss = 
		result +=  (h_function(x_feature, theta) - y[0]) * x_feature[0][i]
		# print(h)
		# print(x_feature, y)
		# loss_j = loss_function_j(x, y) * x
	# print("type result: ", type(result))

	return (1.0 / m) * result


def update_theta(X_train, Y_train, theta, alpha, order_of_regression):
	temp = []
	print("theta: {}".format(len(theta)))

	for i, old_theta in enumerate(theta[0]):

		# print("old_theta: {} derivative_theta_i: {}".format(old_theta, derivative_theta_i(X_train, Y_train, theta, order_of_regression, i)))

		new_theta_i = old_theta - alpha * derivative_theta_i(X_train, Y_train, theta, order_of_regression, i)
		# print("new_theta_i: {}".format(new_theta_i))
		temp.append(new_theta_i)

	# print(np.atleast_2d(temp))
	# exit(0)
	return np.atleast_2d(temp)

"""
Gradient Descent For Multivariate Regression
"""
def gradient_descent(X_train, Y_train, tolerance = 0.001, alpha = 0.001, theta=[], order_of_regression=2):
	# if len(theta) == 0:
	# 	theta = np.atleast_2d(np.random.random(order_of_regression + 1))

	# print(theta.shape)
	# print(theta)

	theta = np.array([[0.89088266, 0.92116762, 0.53932061]])

	loss_history = []
	theta_history = []
	cur_loss = loss_function_j(X_train, Y_train, theta, order_of_regression)
	prev_loss = 0.0
	iter_number = 0
	# return theta

	while True:

		theta = update_theta(X_train, Y_train, theta, alpha, order_of_regression)
		# loss_j = 
		prev_loss = cur_loss
		cur_loss = loss_function_j(X_train, Y_train, theta, order_of_regression)
		iter_number += 1
		print("iter: {} prev_loss: {} cur_loss: {} theta: {}".format(iter_number, prev_loss, cur_loss, theta))
		# break
		if abs(cur_loss - prev_loss) < tolerance:
			break

	# print("cur_loss: {}".format(cur_loss))
	return theta

"""
Plot the line using theta_values
"""
def plot_line(formula, x_range, order_of_regression):
    x = np.array(x_range).tolist()  
    y = [formula(update_features(x_i, order_of_regression)) for x_i in x]
    plt.plot(x, y)


# print(X_train)

theta = gradient_descent(X_train, Y_train)
# result = derivative_theta_i(X_train, Y_train, theta=np.array([[0.5, 0.2, 0.3]]), order_of_regression=2, i=1)
# print(result)
print(theta)


	
def prediction(X_train, theta, order_of_regression=2):
	y_preds = []
	for x in X_train:
		x_feature = update_features(x, order_of_regression)
		# print(value)	
		y_pred = h_function(x_feature, theta)
		print(x[0], x_feature, y_pred)
		y_preds.append(y_pred)
	
	print(y_preds)

	return y_preds

import matplotlib.pyplot as plt

# x = np.array(data).T[0]
# y = np.array(data).T[1]
x = X_train.reshape(-1, 1)
y = Y_train.reshape(-1, 1)

print(X_train.shape)
y_preds = prediction(X_train, theta)
# x, y = zip(*np.array(X_train))

# y_preds = prediction(x, theta)
# # print(x, y)
plt.scatter(x, y, color='red')
plt.scatter(x, y_preds, color='blue')

plt.show()