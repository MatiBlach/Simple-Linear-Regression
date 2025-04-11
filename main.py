import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

ones = np.ones((len(x_train)))
X = np.column_stack((ones,x_train))

# TODO: calculate closed-form solution
theta_best = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.transpose(X)),y_train)



# TODO: calculate error
predictions = theta_best[0] + theta_best[1] * x_test
MSE = 0
for i in range(len(x_test)):
    MSE += (y_test[i] - predictions[i]) ** 2
MSE /= len(x_test)
print("LOGS:")
print(theta_best)
print(MSE)


# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
avg_x = np.average(x_train)
avg_y = np.average(y_train)
stdDev_x = np.std(x_train)
stdDev_y = np.std(y_train)
std_x = (x_train - avg_x) / stdDev_x
std_y = (y_train - avg_y) / stdDev_y
std_x_test = (x_test - avg_x) / stdDev_x
std_y_test = (y_test - avg_y) / stdDev_y
Xstd = np.column_stack((ones,std_x))
print()


# TODO: calculate theta using Batch Gradient Descent
gradTheta = np.random.rand(2)
learningRate = 0.001
iterations = 3000

for i in range(iterations):
    gradTheta = gradTheta - 2 / len(Xstd) * learningRate * np.dot(np.transpose(Xstd), (np.dot(Xstd, gradTheta)) - std_y)
    
    

gradTheta[1] = gradTheta[1] * stdDev_y / stdDev_x
gradTheta[0] = avg_y - gradTheta[1] * avg_x

# TODO: calculate error 

gradPredictions = gradTheta[0] + gradTheta[1] * x_test
gradMSE = 0
for i in range(len(x_test)):
    gradMSE += (y_test[i] - gradPredictions[i]) ** 2
gradMSE /= len(x_test)


print("LOGS:")
print(gradTheta)
print(gradMSE)

# plot the regression line (standarized theta and datasets)
x = np.linspace(min(x_test), max(x_test), 100)
y = float(gradTheta[0]) + float(gradTheta[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()



