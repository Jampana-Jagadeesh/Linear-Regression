# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Read training data from a CSV file
train_data = pd.read_csv("/content/drive/MyDrive/Linear_regression/Linear_regression_train.csv")

# Extract the 'x' feature for training
x_train = train_data[['x']]

# Extract the target variable 'y' for training
y_train = train_data['y']

# Read test data from a CSV file
test_data = pd.read_csv("/content/drive/MyDrive/Linear_regression/Linear_regression_test.csv")

# Extract the 'x' feature for testing
x_test = test_data['x']

# Extract the target variable 'y' for testing
y_test = test_data['y']

# Create a Linear Regression model
model = LinearRegression()

# Fit the model using the training data
model.fit(x_train, y_train)

# Predict 'y' for the training data
y_train_pred = model.predict(x_train)

# Reshape the test data for prediction
x_test = x_test.values.reshape(-1, 1)

# Predict 'y' for the test data
y_test_pred = model.predict(x_test)
print(y_test_pred)

# Calculate Mean Squared Error for the training set
mse_train = mean_squared_error(y_train, y_train_pred)

# Calculate R^2 Score for the training set
r2_train = r2_score(y_train, y_train_pred)

# Calculate Mean Squared Error for the test set
mse_test = mean_squared_error(y_test, y_test_pred)

# Calculate R^2 Score for the test set
r2_test = r2_score(y_test, y_test_pred)

# Print results for the training set
print("Training Set:")
print(f"Mean Squared Error: {mse_train:.2f}")
print(f"R^2 Score: {r2_train:.2f}")

# Print results for the test set
print("\nTest Set:")
print(f"Mean Squared Error: {mse_test:.2f}")
print(f"R^2 Score: {r2_test:.2f}")

# Plot the actual test data points
plt.scatter(x_test, y_test, color='black', label="Actual data")

# Plot the regression line using predicted values
plt.plot(x_test, y_test_pred, color='red', linewidth=3, label="Regression Line")

# Set labels and title for the plot
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression - Test Set')

# Display the legend
plt.legend()

# Show the plot
plt.show()
