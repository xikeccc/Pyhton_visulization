import matplotlib.pyplot as plt
import pandas as pd


exp_df = pd.read_excel('calibration_Gini1.xlsx')
theo_df = pd.read_excel('calibration_Gini1.xlsx')
experimental_error = exp_df['Experimental error rate']
theoretical_error = theo_df['Theoretical error rate']

x = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
y = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


plt.plot(theoretical_error, experimental_error, marker='o', linestyle='-', color='blue', label='Experimental error rate')
plt.plot(x, y, marker='o', linestyle='-', color='orange', label='Theoretical error rate')

plt.xlabel('Expected error rate', fontsize=14)
plt.ylabel('Real error rate', fontsize=14)
plt.title('Error rates comparison', fontsize=16)
plt.legend()
plt.grid(False)
plt.show()


exp_df = pd.read_excel('calibration_GA1.xlsx')
theo_df = pd.read_excel('calibration_GA1.xlsx')
experimental_error = exp_df['Experimental error rate']
theoretical_error = theo_df['Theoretical error rate']

x = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
y = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

plt.plot(theoretical_error, experimental_error, marker='o', linestyle='-', color='blue', label='Experimental error rate')
plt.plot(x, y, marker='o', linestyle='-', color='orange', label='Theoretical error rate')

plt.xlabel('Expected error rate', fontsize=14)
plt.ylabel('Real error rate', fontsize=14)
plt.title('Error rates comparison', fontsize=16)
plt.legend()
plt.grid(False)
plt.show()