import numpy as np
import matplotlib.pyplot as plt

# Define input values
x = np.linspace(-5, 5, 100)

# ReLU activation function
relu = np.maximum(0, x)

# ELU activation function with α = 1 (you can adjust α as needed)
alpha = 1
elu = np.where(x >= 0, x, alpha * (np.exp(x) - 1))

# Plotting ReLU
plt.figure(figsize=(8, 6))
plt.plot(x, relu, label='ReLU')
plt.xlabel('Input', fontsize=14)
plt.ylabel('Output', fontsize=14)
plt.title('ReLU Activation Function', fontsize=16)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()
plt.grid(False)
plt.show()

# Plotting ELU
plt.figure(figsize=(8, 6))
plt.plot(x, elu, label='ELU (α = 1)')
plt.xlabel('Input', fontsize=14)
plt.ylabel('Output', fontsize=14)
plt.title('ELU Activation Function', fontsize=16)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()
plt.grid(False)
plt.show()
