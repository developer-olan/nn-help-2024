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
from keras.layers import Input
from keras.models import Model
import scipy
from scipy.linalg import norm
from scipy.stats import norm
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
import random
from scipy import linalg
from numpy import mat
import tensorflow 
from matplotlib import pyplot
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers


# In[ ]:


# Normalization the data
df1 = pd.read_excel('/home/hashemi/Fei Shao/new_data/new_train.xlsx', dtype=np.float32)
y_train = df1[['Static_Fric']]
x_train = df1[['Size','Angle_left','Angle_right','Restitution','Density','Speed','Rolling_Fric']]
mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std

df2 = pd.read_excel('/home/hashemi/Fei Shao/new_data/test_new1.xlsx', dtype=np.float32)
y_test = df2[['Static_Fric']].values
x_test = df2[['Size','Angle_left','Angle_right','Restitution','Density','Speed','Rolling_Fric']].values
mean = x_test.mean(axis=0)
x_test -= mean
std = x_test.std(axis=0)
x_test /= std


# In[ ]:


train_data = pd.concat([x_train,y_train], axis=1, join="outer", ignore_index=False)


# In[ ]:


# creating generator with MLP
def build_generator():
    model = Sequential()
    model.add(Dense(units=7, input_dim=7))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25))
    model.add(Dense(units=32))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25))
    model.add(Dense(units=32))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25))
    model.add(Dense(units=1))
    return model
generator = build_generator()


# In[ ]:


def build_discriminator():
    
    model = Sequential()
    model.add(Dense(units=128, input_dim=1))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25))

    model.add(Dense(units=64))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25))

    model.add(Dense(units=16))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25))
    model.add(Dense(units=1, activation='sigmoid'))
    
    # compile the model
    sgd = optimizers.SGD(learning_rate=0.000005, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model
discriminator = build_discriminator()


# In[ ]:


def get_accuracy(y_train,generated_data):
    n = y_train.shape[0]
    acc_tem = K.abs(y_train - generated_data) / y_train
    Acc_static_fric = (1 - K.sqrt(K.sum(acc_tem * acc_tem) / n)) * 100
    return Acc_static_fric


# In[ ]:


def build_GAN(discriminator, generator):
    
    discriminator.trainable=False
    GAN_input = Input(shape=(7,))
    x = generator(GAN_input)
    GAN_output= discriminator(x)
    GAN = Model(inputs=GAN_input, outputs=GAN_output)
    # compile model
    adam = Adam(learning_rate=0.0001, beta_1=0.5)
    GAN.compile(loss='binary_crossentropy', optimizer=adam, metrics=[get_accuracy])
    return GAN
  
GAN = build_GAN(discriminator, generator)


# In[ ]:


def plt_results():
    # results before log function
    generated_data = generator.predict(x_test)
    times = generated_data / y_test
    mean = times.mean()
    std = times.std()
    num_bins = 100
    n, bins, patches = plt.hist(times, num_bins, density=True)
    y = norm.pdf(bins, mean, std)
    plt.plot(bins, y, label = 'Static_Fric', color='r')
    plt.xlabel('Deviation')
    plt.ylabel('Probability')
    plt.title('Deviation of predictions for Static friction')
    plt.legend()
    plt.savefig('/home/hashemi/Fei Shao/GAN results/pre before train.png')
    plt.show()
    
    
    # results after log function
    Static_Fric = times
    times[times<0] = 0.01
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
    plt.savefig('/home/hashemi/Fei Shao/GAN results/pre after train.png')
    plt.show()
    
    from matplotlib import pyplot
    pyplot.scatter(generated_data, y_test, color='g')
    pyplot.scatter(y_test, y_test, color='r')
    plt.xlabel('generated_data')
    plt.ylabel('y_test')
    plt.title('Compare predicted value with true value for static friction after training')
    plt.savefig('/home/hashemi/Fei Shao/GAN results/compare.png')
    pyplot.show()


# In[ ]:


def train_GAN(epochs,batch_size):
    # creating GAN
    generator = build_generator()
    discriminator = build_discriminator()
    GAN = build_GAN(discriminator, generator)
    
    # creat list for plt
    d_loss_list = []
    d_fake_loss_list = []
    d_real_loss_list = []
    d_acc_list = []
    
    g_loss_list = []
    fake_data_list = []
    g_acc_list = []
    
    
    # times to train discriminator
    k = 5
 
    for i in range(1, epochs+1): 
        for _ in tqdm(range(batch_size)):
            # generate the fake data
            train_random = train_data.sample(n = batch_size, axis = 0)
            noise = train_random[['Size','Angle_left','Angle_right','Restitution','Density','Speed','Rolling_Fric']].values
            fake_data = generator.predict(noise)
            
            # Select real data from dataset
            y_train = train_random[['Static_Fric']].values
            real_data = y_train
        
            # Labels for fake and real data         
            label_fake = np.zeros(batch_size)
            label_real = np.ones(batch_size)
        
            # update discriminator
            for _ in range(1, k+1):  
                d_real_states = discriminator.train_on_batch(real_data, label_real)
                d_fake_states = discriminator.train_on_batch(fake_data, label_fake)
                #d_states = d_real_states + d_fake_states
                d_states = 0.5 * np.add(d_real_states, d_fake_states)
                 
            # Train the generator/chained GAN model with frozen weights in discriminator
            train_random2 = train_data.sample(n = batch_size, axis = 0)
            noise2 = train_random2[['Size','Angle_left','Angle_right','Restitution','Density','Speed','Rolling_Fric']].values
            
            # create inverted labels for the fake samples
            y_gan = np.ones(batch_size)
            # update the generator via the discriminator's error
            g_states = GAN.train_on_batch(noise2, y_gan)
        
            
        # data to plot
        d_loss_list.append(d_states[0])  # d_states[0] means discriminator loss
        d_fake_loss_list.append(d_fake_states[0])
        d_real_loss_list.append(d_real_states[0])
        d_acc_list.append(d_states[1])   # d_states[1] means adiscriminator accuracy
             
        g_loss_list.append(g_states[0])
        fake_data_list.append(fake_data)
        g_acc_list.append(g_states[1])
            
    plt_results()
    
    # plot the loss of discriminator and generator
    plt.plot(d_loss_list, label='d_loss')
    plt.plot(g_loss_list, label='g_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['d_loss','g_loss'])
    plt.savefig('/home/hashemi/Fei Shao/GAN results/loss.png')
    plt.show()
    
    # plot the loss of real data and fake data for discriminator
    plt.plot(d_fake_loss_list, label='d_fake_loss')
    plt.plot(d_real_loss_list, label='d_real_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['d_fake_loss','d_real_loss'])
    plt.savefig('/home/hashemi/Fei Shao/GAN results/loss of real data and fake data.png')
    plt.show()
    
   # plot the accuracy of discriminator
    plt.plot(d_acc_list, label='d_acc')
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['d_acc'])
    plt.savefig('/home/hashemi/Fei Shao/GAN results/acc d.png')
    plt.show()
    
    plt.plot(d_real_loss_list, label='c_real_loss')
    plt.plot(d_fake_loss_list, label='c_fake_loss')
    plt.plot(g_loss_list, label='g_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['c_real_loss','c_fake_loss','g_loss'])
    plt.savefig('/home/hashemi/Fei Shao/GAN results/all_loss.png')
    plt.show()
    
    # plot the accuracy of generator
    plt.plot(g_acc_list, label='g_acc')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['g_acc'])
    plt.show()  


# In[ ]:


train_GAN(epochs=120, batch_size=128) #k=5

