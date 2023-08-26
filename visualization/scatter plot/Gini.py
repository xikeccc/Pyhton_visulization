import pandas as pd
import matplotlib.pyplot as plt

############        YSI
training_df = pd.read_excel('YSI_gini.xlsx', sheet_name='training')
test_df = pd.read_excel('YSI_gini.xlsx', sheet_name='test')

prediction_train = training_df['dense_5_0:0_0']
actual_train = training_df['Column0']
prediction_test = test_df['dense_5_0:0_0']
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


############        CN
training_df = pd.read_excel('CN_gini.xlsx', sheet_name='training')
test_df = pd.read_excel('CN_gini.xlsx', sheet_name='test')

prediction_train = training_df['dense_5_0:0_0']
actual_train = training_df['Column0']
prediction_test = test_df['dense_5_0:0_0']
actual_test = test_df['Column0']

plt.figure(figsize=(10, 6))
plt.scatter(prediction_train, actual_train, color='blue', label='Training Set')
plt.scatter(prediction_test, actual_test, color='red', label='Test Set')
plt.xlabel('Predicted CN', fontsize=14)
plt.ylabel('Actual CN', fontsize=14)
plt.title('CN', fontsize=16)
plt.legend()
plt.grid(False)
plt.show()

############        KV
training_df = pd.read_excel('KV_gini.xlsx', sheet_name='training')
test_df = pd.read_excel('KV_gini.xlsx', sheet_name='test')

prediction_train = training_df['dense_5_0:0_0']
actual_train = training_df['Column0']
prediction_test = test_df['dense_5_0:0_0']
actual_test = test_df['Column0']

plt.figure(figsize=(10, 6))
plt.scatter(prediction_train, actual_train, color='blue', label='Training Set')
plt.scatter(prediction_test, actual_test, color='red', label='Test Set')
plt.xlabel('Predicted KV', fontsize=14)
plt.ylabel('Actual KV', fontsize=14)
plt.title('KV', fontsize=16)
plt.legend()
plt.grid(False)
plt.show()


############        MON
training_df = pd.read_excel('MON_gini.xlsx', sheet_name='training')
test_df = pd.read_excel('MON_gini.xlsx', sheet_name='test')

prediction_train = training_df['dense_5_0:0_0']
actual_train = training_df['Column0']
prediction_test = test_df['dense_5_0:0_0']
actual_test = test_df['Column0']

plt.figure(figsize=(10, 6))
plt.scatter(prediction_train, actual_train, color='blue', label='Training Set')
plt.scatter(prediction_test, actual_test, color='red', label='Test Set')
plt.xlabel('Predicted MON', fontsize=14)
plt.ylabel('Actual MON', fontsize=14)
plt.title('MON', fontsize=16)
plt.legend()
plt.grid(False)
plt.show()


############       IT
training_df = pd.read_excel('IT_gini.xlsx', sheet_name='training')
test_df = pd.read_excel('IT_gini.xlsx', sheet_name='test')

prediction_train = training_df['dense_5_0:0_0']
actual_train = training_df['Column0']
prediction_test = test_df['dense_5_0:0_0']
actual_test = test_df['Column0']

plt.figure(figsize=(10, 6))
plt.scatter(prediction_train, actual_train, color='blue', label='Training Set')
plt.scatter(prediction_test, actual_test, color='red', label='Test Set')
plt.xlabel('Predicted IT', fontsize=14)
plt.ylabel('Actual IT', fontsize=14)
plt.title('IT', fontsize=16)
plt.legend()
plt.grid(False)
plt.show()



