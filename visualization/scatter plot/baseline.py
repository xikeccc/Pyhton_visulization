import pandas as pd
import matplotlib.pyplot as plt

############       baseline
training_df = pd.read_excel('YSI_baseline.xlsx', sheet_name='train')
test_df = pd.read_excel('YSI_baseline.xlsx', sheet_name='test')

prediction_train = training_df['dense_3_0:0_0']
actual_train = training_df['Column0']
prediction_test = test_df['dense_3_0:0_0']
actual_test = test_df['Column0']

plt.figure(figsize=(10, 6))
plt.scatter(prediction_train, actual_train, color='blue', label='Training Set')
plt.scatter(prediction_test, actual_test, color='red', label='Test Set')
plt.xlabel('Predicted YSI', fontsize=14)
plt.ylabel('Actual YSI', fontsize=14)
plt.title('YSI', fontsize=16)
plt.legend()
plt.grid(False)
plt.show()