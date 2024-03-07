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
from sklearn.preprocessing import MinMaxScaler
import random
from scipy import linalg
from numpy import mat
import tensorflow 
from matplotlib import pyplot
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from keras import backend
from pylab import *
from keras.constraints import Constraint
from keras.initializers import RandomNormal
from keras.models import load_model


# In[ ]:


# read data
train_data = pd.read_excel('/home/hashemi/Fei Shao/total.xlsx', dtype=np.float32)
#train_data = pd.read_excel('/home/hashemi/Fei Shao/total.xlsx', dtype=np.float32, nrows=2705)
train_data = train_data[2706:5104]


# In[ ]:


# clip model weights to a given hypercube
class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value
 
    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)
 
    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


# In[ ]:


# implementation of wasserstein loss
def wasserstein_loss(real_data, fake_data):
    return backend.mean(real_data * fake_data)


# In[ ]:


# creating generator with MLP
def build_generator():
    # weight initialization
    init = RandomNormal(stddev=0.02)

    model = Sequential()
    model.add(Dense(units=64, kernel_initializer=init, input_dim=8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Dense(units=32, kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Dense(units=16, kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Dense(units=8, kernel_initializer=init))
    return model

generator = build_generator()


# In[ ]:


def build_discriminator():
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # weight constraint
    const = ClipConstraint(0.01)  
    
    model = Sequential()
    model.add(Dense(units=128, kernel_initializer=init, kernel_constraint=const, input_dim=8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    model.add(Dense(units=64, kernel_initializer=init, kernel_constraint=const))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Dense(units=32, kernel_initializer=init, kernel_constraint=const))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    model.add(Dense(units=16, kernel_initializer=init, kernel_constraint=const))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    model.add(Dense(units=1))
    # compile the model
    opt = RMSprop(learning_rate=0.0001) 
    model.compile(loss=wasserstein_loss, optimizer=opt, metrics=['accuracy'])
    return model

discriminator = build_discriminator()


# In[ ]:


def build_GAN(discriminator, generator):
    
    discriminator.trainable=False
    GAN_input = Input(shape=(8,))
    x = generator(GAN_input)
    GAN_output = discriminator(x)
    GAN = Model(inputs=GAN_input, outputs=GAN_output)
    # compile model
    opt = RMSprop(learning_rate=0.0001)
    GAN.compile(loss=wasserstein_loss, optimizer=opt)
    return GAN
  
GAN = build_GAN(discriminator, generator)


# In[ ]:


def generate_data(generator, batch_size):
    # generate fake data
    noise = np.random.normal(loc=0, scale=1, size=(batch_size, 8))
    generated_data = generator.predict(noise)

    # save generated_data(fake data) into Excel
    data = pd.DataFrame(generated_data)
    data.to_excel('/home/hashemi/Fei Shao/fake_4.5_300epoch.xlsx')
    
    x = np.arange(0, len(generated_data)) #x-axis coordinate for fake data
    z = np.arange(0, batch_size)          #x-axis coordinate for real data 
    #figure(8)
    #subplot(8,2,1)
    pyplot.scatter(x, generated_data[:,0]/100, color='g',alpha=0.4)
    pyplot.scatter(z, train_data[['Size']].sample(n = batch_size, axis = 0).values, color='r')
    plt.ylabel('Size')
    plt.title('Compare predicted value with true value after training')
    plt.savefig('/home/hashemi/Fei Shao/GAN results/Size.png')
    pyplot.show()
    
    #subplot(8,2,2)
    pyplot.scatter(x, generated_data[:,1], color='g',alpha=0.4)
    pyplot.scatter(z, train_data[['Angle_left']].sample(n = batch_size, axis = 0).values, color='r')
    plt.ylabel('Angle_left')
    plt.title('Compare predicted value with true value after training')
    plt.legend(['generated_data','real_data'])
    plt.savefig('/home/hashemi/Fei Shao/GAN results/Angle_left.png')
    pyplot.show()
    
    #subplot(8,2,3)
    pyplot.scatter(x, generated_data[:,2], color='g',alpha=0.4)
    pyplot.scatter(z, train_data[['Angle_right']].sample(n = batch_size, axis = 0).values, color='r')
    plt.ylabel('Angle_right')
    plt.title('Compare predicted value with true value after training')
    plt.legend(['generated_data','real_data'])
    plt.savefig('/home/hashemi/Fei Shao/GAN results/Angle_right.png')
    pyplot.show()
    
    #subplot(8,2,4)
    pyplot.scatter(x, generated_data[:,3]/100000, color='g',alpha=0.4)
    pyplot.scatter(z, train_data[['Restitution']].sample(n = batch_size, axis = 0).values, color='r')
    plt.ylabel('Restitution')
    plt.title('Compare predicted value with true value after training')
    plt.legend(['generated_data','real_data'])
    plt.savefig('/home/hashemi/Fei Shao/GAN results/Restitution.png')
    pyplot.show()
    
    #subplot(8,2,5)
    pyplot.scatter(x, generated_data[:,4]*10, color='g',alpha=0.4)
    pyplot.scatter(z, train_data[['Density']].sample(n = batch_size, axis = 0).values, color='r')
    plt.ylabel('Density')
    plt.title('Compare predicted value with true value after training')
    plt.legend(['generated_data','real_data'])
    plt.savefig('/home/hashemi/Fei Shao/GAN results/Density.png')
    pyplot.show()
    
    #subplot(8,2,6)
    pyplot.scatter(x, generated_data[:,5]/100, color='g',alpha=0.4)
    pyplot.scatter(z, train_data[['Speed']].sample(n = batch_size, axis = 0).values, color='r')
    plt.ylabel('Speed')
    plt.title('Compare predicted value with true value after training')
    plt.legend(['generated_data','real_data'])
    plt.savefig('/home/hashemi/Fei Shao/GAN results/Speed.png')
    pyplot.show()
    
    #subplot(8,2,7)
    pyplot.scatter(x, generated_data[:,6]/10000, color='g',alpha=0.4)
    pyplot.scatter(z, train_data[['Static_Fric']].sample(n = batch_size, axis = 0).values, color='r')
    plt.ylabel('Static_Fric')
    plt.title('Compare predicted value with true value after training')
    plt.legend(['generated_data','real_data'])
    plt.savefig('/home/hashemi/Fei Shao/GAN results/Static_Fric.png')
    pyplot.show()
    
    #subplot(8,2,8)
    pyplot.scatter(x, generated_data[:,7]/10000, color='g',alpha=0.4)
    pyplot.scatter(z, train_data[['Rolling_Fric']].sample(n = batch_size, axis = 0).values, color='r')
    plt.ylabel('Rolling_Fric')
    plt.title('Compare predicted value with true value after training')
    #plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.2)
    plt.legend(['generated_data','real_data'])
    plt.savefig('/home/hashemi/Fei Shao/GAN results/Rolling_Fric.png')
    pyplot.show()


# In[ ]:


def train_GAN(epochs, batch_size):
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
    #g_acc_list = []
    
    #acc_real_list=[]
    #acc_fake_list=[]
    
    # times to train discriminator
    k = 5
    
    # calculate the size of half a batch of samples
    half_batch = int(batch_size / 2)
    
    for i in range(1, epochs+1): 
        for _ in tqdm(range(batch_size)):
            # generate the fake data
            noise = np.random.normal(loc=0, scale=1, size=(half_batch, 8))
            fake_data = generator.predict(noise)
            
            # Select real data from dataset
            train_random = train_data.sample(n = half_batch, axis = 0)
            train_random.iloc[:,0] = train_random.iloc[:,0]*100
            train_random.iloc[:,3] = train_random.iloc[:,3]*100000
            train_random.iloc[:,4] = train_random.iloc[:,4]/10
            train_random.iloc[:,5] = train_random.iloc[:,5]*100
            train_random.iloc[:,6] = train_random.iloc[:,6]*10000
            train_random.iloc[:,7] = train_random.iloc[:,7]*10000
            real_data = train_random.values
        
            # Labels for fake and real data         
            label_fake = ones((half_batch, 1))
            label_real = -ones((half_batch, 1))
        
            # update discriminator
            for _ in range(1, k+1):  
                d_real_states = discriminator.train_on_batch(real_data, label_real)
                d_fake_states = discriminator.train_on_batch(fake_data, label_fake)
                #d_states = d_real_states + d_fake_states
                d_states = 0.5 * np.add(d_real_states, d_fake_states)
                 
            # Train the generator/chained GAN model with frozen weights in discriminator
            noise2 = np.random.normal(loc=0, scale=1, size=(batch_size, 8))
            # create inverted labels for the fake samples
            y_gan = -ones((batch_size, 1))
            # update the generator via the discriminator's error
            g_states = GAN.train_on_batch(noise2, y_gan)
        
        # data to plot
        d_loss_list.append(d_states[0])  # d_states[0] means discriminator loss
        d_fake_loss_list.append(d_fake_states[0])
        d_real_loss_list.append(d_real_states[0])
        d_acc_list.append(d_states[1])   # d_states[1] means adiscriminator accuracy
             
        g_loss_list.append(g_states)
        #fake_data_list.append(fake_data)
        #g_acc_list.append(g_states[1])
        
        # generate data in the last epoches     
        if i == epochs:
            generate_data(generator, batch_size)
            generator.save('/home/hashemi/Fei Shao/WGAN_2.5_100epoch.h5')
        
    # plot the loss of discriminator and generator
    plt.plot(d_loss_list, label='c_loss')
    plt.plot(g_loss_list, label='g_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['c_loss','g_loss'])
    plt.savefig('/home/hashemi/Fei Shao/GAN results/loss.png')
    plt.show()
    
    # plot the loss of real data and fake data for discriminator
    plt.plot(d_fake_loss_list, label='c_fake_loss')
    plt.plot(d_real_loss_list, label='c_real_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['c_fake_loss','c_real_loss'])
    plt.savefig('/home/hashemi/Fei Shao/GAN results/loss of real data and fake data.png')
    plt.show()
    
    plt.plot(d_real_loss_list, label='c_real_loss')
    plt.plot(d_fake_loss_list, label='c_fake_loss')
    plt.plot(g_loss_list, label='g_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['c_real_loss','c_fake_loss','g_loss'])
    plt.savefig('/home/hashemi/Fei Shao/GAN results/all_loss.png')
    plt.show()
    
   # plot the accuracy of discriminator
    plt.plot(d_acc_list, label='c_acc')
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(['c_acc'])
    plt.savefig('/home/hashemi/Fei Shao/GAN results/acc d.png')
    plt.show()


# In[ ]:


train_GAN(epochs=100, batch_size=128) #k=5 size=2.5

