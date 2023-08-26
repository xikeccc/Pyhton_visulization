from ecnet.datasets import load_cn, load_mon, load_kv, load_mp
from ecnet.datasets.load_data import load_it, load_ysi
from ecnet.tasks.feature_selection import select_rfr
from ecnet.tasks.parameter_tuning import tune_batch_size, tune_model_architecture,\
    tune_training_parameters
from ecnet import ECNet
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import median_absolute_error, r2_score
import torch
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from math import sqrt

dataset = load_ysi(as_dataset=True, backend='alvadesc')
print(type(dataset), dataset.desc_vals.shape, dataset.target_vals.shape)

index_train, index_test = train_test_split([i for i in range(len(dataset))],
                                           test_size=0.2, random_state=42)
dataset_train = deepcopy(dataset)
dataset_train.set_index(index_train)
dataset_test = deepcopy(dataset)
dataset_test.set_index(index_test)
print(dataset_train.desc_vals.shape, dataset_test.desc_vals.shape)

desc_idx, desc_imp = select_rfr(dataset_train, total_importance=0.95,
                                n_estimators=100, n_jobs=4)
dataset_train.set_desc_index(desc_idx)
dataset_test.set_desc_index(desc_idx)
desc_names = [dataset.desc_names[i] for i in desc_idx]
print(dataset_train.desc_vals.shape, dataset_test.desc_vals.shape)
print(desc_names[:5], len(desc_names))

model = ECNet(dataset_train.desc_vals.shape[1], 1, 32, 4)
train_loss, valid_loss = model.fit(
    dataset=dataset_train, valid_size=0.2,verbose=5,
    patience=50, epochs=100, random_state=24
)

y_hat_train = model(dataset_train.desc_vals).detach().numpy()
y_train = dataset_train.target_vals
train_mae = median_absolute_error(y_hat_train, y_train)
train_r2 = r2_score(y_hat_train, y_train)
y_hat_test = model(dataset_test.desc_vals).detach().numpy()
y_test = dataset_test.target_vals
test_mae = median_absolute_error(y_hat_test, y_test)
test_r2 = r2_score(y_hat_test, y_test)
print('Training median absolute error: {}'.format(train_mae))
print('Training r-squared coefficient: {}'.format(train_r2))
print('Testing median absolute error: {}'.format(test_mae))
print('Testing r-squared coefficient: {}'.format(test_r2))

train_loss = [sqrt(l) for l in train_loss][5:]
valid_loss = [sqrt(l) for l in valid_loss][5:]
epoch = [i for i in range(len(train_loss))]
plt.clf()
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Sqrt(Loss)', fontsize=14)
plt.plot(epoch, train_loss, color='blue', label='Training Loss')
plt.plot(epoch, valid_loss, color='red', label='Validation Loss')
plt.legend(loc='upper right')
plt.grid(False)
plt.show()

