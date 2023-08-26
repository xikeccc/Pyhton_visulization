import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Load data from the Excel file
df = pd.read_excel('descriptor_correlation.xlsx')
x = df['SM5_B(v)']
y = df['Column0']

# Define the exponential function to fit
def exponential_function(x, a, b, c):
    return a * np.exp(b * x) + c

# Fit the exponential function to the data
params, covariance = curve_fit(exponential_function, x, y)

# Create a scatter plot
plt.scatter(x, y, color='blue', label='Data Points')

# Generate the fitted curve
x_fit = np.linspace(min(x), max(x), 100)
y_fit = exponential_function(x_fit, *params)

# Plot the fitted exponential curve
plt.plot(x_fit, y_fit, color='red', label=f'Exponential Fit: {params[0]:.4f} * exp({params[1]:.4f} * x) + ({params[2]:.4f})')

# Add labels and title
plt.xlabel('SM5_B(v)')
plt.ylabel('YSI')
plt.title('Scatter Plot with Exponential Fit')
plt.legend()
plt.grid(False)
plt.show()
print(params)

############################################
df = pd.read_excel('descriptor_correlation.xlsx')
x = df['SM6_B(v)']
y = df['Column0']
params, covariance = curve_fit(exponential_function, x, y)

plt.scatter(x, y, color='blue', label='Data Points')
x_fit = np.linspace(min(x), max(x), 100)
y_fit = exponential_function(x_fit, *params)

plt.plot(x_fit, y_fit, color='red', label=f'Exponential Fit: {params[0]:.4f} * exp({params[1]:.4f} * x) + ({params[2]:.4f})')
plt.xlabel('SM6_B(v)')
plt.ylabel('YSI')
plt.title('Scatter Plot with Exponential Fit')
plt.legend()
plt.grid(False)
plt.show()
print(params)

############################################
df = pd.read_excel('descriptor_correlation.xlsx')
x = df['SM4_B(p)']
y = df['Column0']
params, covariance = curve_fit(exponential_function, x, y)

plt.scatter(x, y, color='blue', label='Data Points')
x_fit = np.linspace(min(x), max(x), 100)
y_fit = exponential_function(x_fit, *params)

plt.plot(x_fit, y_fit, color='red', label=f'Exponential Fit: {params[0]:.4f} * exp({params[1]:.4f} * x) + ({params[2]:.4f})')
plt.xlabel('SM4_B(p)')
plt.ylabel('YSI')
plt.title('Scatter Plot with Exponential Fit')
plt.legend()
plt.grid(False)
plt.show()
print(params)

############################################
df = pd.read_excel('descriptor_correlation.xlsx')
x = df['SM5_B(p)']
y = df['Column0']
params, covariance = curve_fit(exponential_function, x, y)

plt.scatter(x, y, color='blue', label='Data Points')
x_fit = np.linspace(min(x), max(x), 100)
y_fit = exponential_function(x_fit, *params)

plt.plot(x_fit, y_fit, color='red', label=f'Exponential Fit: {params[0]:.4f} * exp({params[1]:.4f} * x) + ({params[2]:.4f})')
plt.xlabel('SM5_B(p)')
plt.ylabel('YSI')
plt.title('Scatter Plot with Exponential Fit')
plt.legend()
plt.grid(False)
plt.show()
print(params)

############################################
df = pd.read_excel('descriptor_correlation.xlsx')
x = df['SM6_B(p)']
y = df['Column0']
params, covariance = curve_fit(exponential_function, x, y)

plt.scatter(x, y, color='blue', label='Data Points')
x_fit = np.linspace(min(x), max(x), 100)
y_fit = exponential_function(x_fit, *params)

plt.plot(x_fit, y_fit, color='red', label=f'Exponential Fit: {params[0]:.4f} * exp({params[1]:.4f} * x) + ({params[2]:.4f})')
plt.xlabel('SM6_B(p)')
plt.ylabel('YSI')
plt.title('Scatter Plot with Exponential Fit')
plt.legend()
plt.grid(False)
plt.show()
print(params)