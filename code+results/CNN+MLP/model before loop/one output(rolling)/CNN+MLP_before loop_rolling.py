#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
tf.config.list_physical_devices()
import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Input,Flatten
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
import os
import glob
import cv2
import tensorflow as tf
from keras import backend as K
from PIL import Image
from skimage.io import imread, imshow
import matplotlib.mlab as mlab
from scipy.stats import norm
import seaborn as sns


# In[ ]:


# Load Numerical Data
df = pd.read_excel('/home/somayeh/Desktop/Fei/total.xlsx', dtype=np.float32, nrows = 2705)
df = df.drop(labels = ['Angle_left', 'Angle_right','Size'], axis = 1)
df


# In[ ]:


images = glob.glob("/home/somayeh/Desktop/Fei/new_1.5_pic/*.png")
images = sorted(images,key = lambda x: int(x.split('/')[-1].split('.')[0]))

X= []

#load the data
for img in images:
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[60:360, 70:570]                      # Crop coordinates
    scale_percent = 30                                # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim) 
    image = image / 255
    image = image.astype(np.float32)
    X.append(image)


# In[ ]:


from skimage.io import imread, imshow
imshow(X[2704])
plt.show()


# In[ ]:


X = np.array(X)
X.shape


# In[ ]:


#split all data together
num_train, num_test, image_train, image_test = train_test_split(df,
    X,
    test_size=0.2,
    shuffle=True,                                                            
    )
num_train_x=num_train.drop(labels = ['Rolling_Fric'], axis = 1)
num_train_y=num_train.iloc[:,-1:]

num_test_x=num_test.drop(labels = ['Rolling_Fric'], axis = 1)
num_test_y=num_test.iloc[:,-1:]


# In[ ]:


# normalization dataframe
mean1 = num_train_x.mean(axis=0)
num_train_x -= mean1
std1 = num_train_x.std(axis=0)
num_train_x /= std1

mean2 = num_test_x.mean(axis=0)
num_test_x -= mean2
std2 = num_train_x.std(axis=0)
num_test_x /= std2


# In[ ]:


#MLP Model
inputs = Input(shape=(4,))
x = Dense(256, activation="relu")(inputs)
x = Dropout(0.25)(x)
x = Dense(128,activation="relu")(x)
x = Dropout(0.25)(x)
x = Dense(64,activation="relu" )(x)
x = Dense(16,activation="relu" )(x)
x = Dense(2,activation="relu" )(x)
model1 = Model(inputs, x)


# In[ ]:


# build CNN keras model
inputs = Input(shape=(90, 150, 1))  # size of input data(images) 
x = Conv2D(128, (3,3), activation = 'relu', padding='same') (inputs)   
x = MaxPool2D(pool_size =(2,2))(x)
x = Conv2D(64, (3,3), activation = 'relu', padding='same') (x)               
x = MaxPool2D(pool_size =(2,2))(x)
x = Flatten()(x)
x = Dense(64, activation = 'relu')(x)
x = Dense(32, activation = 'relu')(x)
x = Dense(8, activation = 'relu')(x)
x = Dense(2, activation="linear")(x)
CNN_Model = Model(inputs,x)


# In[ ]:


# combine the two model
combinedInput = concatenate([model1.output, CNN_Model.output])
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)
model = Model(inputs=[model1.input, CNN_Model.input], outputs=x)


# In[ ]:


def my_metric(num_test, predict):
    n = len(num_test)
    # convert n int to float32
    n = tf.cast(n, tf.float32)
    acc_tem = K.abs(num_test - predict) / num_test

    acc_right = (1 - K.sqrt(K.sum(acc_tem[:, :] * acc_tem[:, :]) / n)) * 100
    
    return acc_right


# In[ ]:


model.compile(loss='mse', optimizer= 'Adam', metrics=[my_metric])
# report path
report_path = '/home/somayeh/Desktop/Fei/'
checkpoint = ModelCheckpoint(os.path.join(report_path, 'V11rolling.h5'),
                            monitor='val_loss',
                            mode='auto',
                            verbose=1,
                            save_best_only=True
                            )
#train the model
history = model.fit(
x=[num_train_x, image_train], y=num_train_y, 
validation_split=0.04, 
epochs=50, batch_size=128, verbose=1, callbacks=[checkpoint])


# In[ ]:


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('The Loss')
plt.legend()
plt.savefig('/home/somayeh/Desktop/Fei/roll_Loss.png')
plt.show()


# In[ ]:


plt.plot(history.history['my_metric'], label='my_metric')
plt.plot(history.history['val_my_metric'], label='val_my_metric')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('The accuracy')
plt.legend()
plt.savefig('/home/somayeh/Desktop/Fei/roll_acc.png')
plt.show()


# In[ ]:


predict = model.predict([num_test_x, image_test])
plt.plot(predict)


# In[ ]:


times = predict / num_test_y
Roll_Fric = times
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
plt.savefig('/home/somayeh/Desktop/Fei/rolling_Probability.png')
plt.show()

