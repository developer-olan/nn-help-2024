#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
import os
from scipy.linalg import norm
from scipy.stats import norm


# In[ ]:


# report path
report_path = r'/home/hashemi/Fei Shao/fake data/MLP results'

# read data
df = pd.read_excel('/home/hashemi/Fei Shao/total_fake.xlsx', dtype=np.float32)
df = df.drop(df[df.iloc[:,7]>=0.1].index) 
# spilt train and test data
y = df[['Static_Fric', 'Rolling_Fric']].values
x = df.drop(labels=['Static_Fric', 'Rolling_Fric'], axis=1)
x = x.values

# normalization dataframe
mean = x.mean(axis=0)
x -= mean
std = x.std(axis=0)
x /= std

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    shuffle=True
    )


# In[ ]:


def my_metric(y_test1, predict):
    n = len(y_test1)
    # convert n int to float32
    n = tf.cast(n, tf.float32)
    acc_tem = K.abs(y_test1 - predict) / y_test1

    acc_left = (1 - K.sqrt(K.sum(acc_tem[:, 0] * acc_tem[:, 0]) / n)) * 100
    acc_right = (1 - K.sqrt(K.sum(acc_tem[:, 1] * acc_tem[:, 1]) / n)) * 100
    average = (acc_left + acc_right) / 2

    return average


# In[ ]:


# build model
model = Sequential()
model.add(Dense(units=512, activation='relu', input_dim=6))
model.add(Dropout(0.25))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=2))

model.compile(optimizer='adam', loss='mse', metrics=[my_metric])
# show model
#model.summary()

# save best model and callback
checkpoint = ModelCheckpoint(os.path.join(report_path, 'best_model_delete.h5'),
                            monitor='val_loss',
                            mode='auto',
                            verbose=1,
                            save_best_only=True
                            )


# In[ ]:


# train
history = model.fit(x_train,
                    y_train,
                    verbose=1,
                    batch_size=128,
                    epochs=100,
                    validation_split=0.04,
                    callbacks=[checkpoint]
                    )


# In[ ]:


# save loss figure
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('The Loss')
plt.legend()
plt.savefig(os.path.join(report_path, 'loss.png'))
plt.show()

# save acc figure
plt.plot(history.history['my_metric'], label='my_metric')
plt.plot(history.history['val_my_metric'], label='val_my_metric')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.title('The accuracy')
plt.legend()
plt.savefig(os.path.join(report_path, 'accuracy.png'))
plt.show()


# In[ ]:


generated_data = model.predict(x_test)
times = generated_data / y_test
# results after log function
Static_Fric = times[:,0]
Static_Fric[Static_Fric<0] = 0.01
Static_Fric = 10*np.log10(Static_Fric)
mean = Static_Fric.mean()
std = Static_Fric.std()
num_bins = 100
n, bins, patches = plt.hist(Static_Fric, num_bins, density=True)
y = norm.pdf(bins, mean, std)
plt.plot(bins, y, label = 'Static_Fric', color='r')
plt.xlabel('Deviation')
plt.ylabel('Probability')
plt.title('Deviation of predictions for Static friction')
plt.legend()
plt.savefig(os.path.join(report_path, 'Static.png'))
plt.show()

Roll_Fric = times[:,-1]
Roll_Fric[Roll_Fric<0] = 0.01
Roll_Fric = 10*np.log10(Roll_Fric)
mean = Roll_Fric.mean()
std = Roll_Fric.std()
num_bins = 100
n, bins, patches = plt.hist(Roll_Fric, num_bins, density=True)
y = norm.pdf(bins, mean, std)
plt.plot(bins, y, label = 'Roll_Fric', color='r')
plt.xlabel('Deviation')
plt.ylabel('Probability')
plt.title('Deviation of predictions for Rolling friction')
plt.legend()
plt.savefig(os.path.join(report_path, 'Rolling.png'))
plt.show()

