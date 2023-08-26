import pandas as pd
import matplotlib.pyplot as plt

# Load data from the Excel file
excel_file = 'interval_own.xlsx'  # Replace with the actual file path
df = pd.read_excel(excel_file, index_col=0)  # Use the first column as the index

# Create a line plot of prediction and actual values
plt.figure(figsize=(8, 6))
plt.plot(df.index, df['dense_5/Relu:0_0'], color='blue', label='Predicted YSI')
plt.plot(df.index, df['Column0'], color='green', label='Actual YSI')



# Plot prediction intervals as shaded area
plt.fill_between(df.index, df['Lower bound'], df['Upper bound'], color='gray', alpha=0.3, label='Prediction Interval')

# Set labels and title
plt.xlabel('Index', fontsize=14)  # Increased font size
plt.ylabel('Values', fontsize=14)  # Increased font size
plt.title('Prediction and Actual Values with Prediction Interval', fontsize=16)  # Increased font size
plt.legend(fontsize=12)  # Increased legend font size

# Remove grid
plt.grid(False)

plt.show()
