#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from sklearn.model_selection import train_test_split
import os
from keras.models import clone_model
from keras.callbacks import ModelCheckpoint
from scipy.interpolate import make_interp_spline
from keras import backend as K
from scipy.stats import norm

import keras
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Input,Flatten
from keras.models import Sequential, Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
import glob
import cv2
from keras import backend as K
from PIL import Image
from skimage.io import imread, imshow
import matplotlib.mlab as mlab
import seaborn as sns
from keras.models import load_model


# In[ ]:


# creating custom metrics, show with %
def my_metric(y_test1, predict):
    n = len(y_test1)
    # convert n int to float32
    n = tf.cast(n, tf.float32)
    acc_tem = K.abs(y_test1 - predict) / y_test1

    acc_left = (1 - K.sqrt(K.sum(acc_tem[:, 0] * acc_tem[:, 0]) / n)) * 100
    acc_right = (1 - K.sqrt(K.sum(acc_tem[:, 1] * acc_tem[:, 1]) / n)) * 100
    average = (acc_left + acc_right) / 2

    return average

# get best model in n models; acc is a list which stores n ANN's accuracy
def get_best_model(acc, keras_model):

    # get max evaluation value
    eva_max_loc = acc.index(max(acc))

    return keras_model[eva_max_loc]

# for test data 2 to get average accuracy
def get_accuracy(test_data,predict_data):
    n = len(test_data)

    acc_tem = np.abs(test_data - predict_data) / test_data
    # acc_tem[acc_tem > 1] = 0
    Acc_stat_fric = (1 - np.sqrt(np.sum(acc_tem[:, 0] * acc_tem[:, 0]) / n)) * 100
    Acc_roll_fric = (1 - np.sqrt(np.sum(acc_tem[:, 1] * acc_tem[:, 1]) / n)) * 100

    return Acc_stat_fric, Acc_roll_fric


# In[ ]:


# report path
report_path = '/home/somayeh/Desktop/Fei/report_RL/'

# Load Numerical Data
df = pd.read_excel('/home/somayeh/Desktop/Fei/total.xlsx', dtype=np.float32, nrows = 2705)
df = df.drop(labels = ['Angle_left', 'Angle_right','Size'], axis = 1)

images = glob.glob("/home/somayeh/Desktop/Fei/delete_0.3_pic/*.png")
images = sorted(images,key = lambda x: int(x.split('/')[-1].split('.')[0]))

X= []

#load the data
for img in images:
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[60:360, 70:570]                     # Crop coordinates
    scale_percent = 30                                # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim) 
    image = image / 255
    image = image.astype(np.float32)
X.append(image)

imshow(X[2])
plt.show()
X = np.array(X)
X.shape
df.shape

#split all data together
num_train, num_test, image_train, image_test = train_test_split(df,
    X,
    test_size=0.2,
    shuffle=True,
    )

num_train_x=num_train.drop(labels = ['Static_Fric', 'Rolling_Fric'], axis = 1)
num_train_y=num_train.iloc[:,-2:]

num_test_x=num_test.drop(labels = ['Static_Fric', 'Rolling_Fric'], axis = 1)
num_test_y=num_test.iloc[:,-2:]

# normalization dataframe
mean = num_train_x.mean(axis=0)
num_train_x -= mean
std = num_train_x.std(axis=0)
num_train_x /= std

mean1 = num_test_x.mean(axis=0)
num_test_x -= mean1
std1 = num_test_x.std(axis=0)
num_test_x /= std1
num_test = pd.concat([num_test_x,num_test_y], axis=1, join="outer", ignore_index=False)

num_test

# split image_test1 and image_test2, split test1 and test2
num_test1, num_test2, image_test1, image_test2 = train_test_split(
    num_test, image_test,
    test_size=0.5,
    shuffle=True,
    )
x_test1=num_test1.drop(labels = ['Static_Fric', 'Rolling_Fric'], axis = 1).values
y_test1=num_test1.iloc[:,-2:].values

x_test2=num_test2.drop(labels = ['Static_Fric', 'Rolling_Fric'], axis = 1).values
y_test2=num_test2.iloc[:,-2:].values


# In[ ]:


# load model from keras best selected
# copy0 means the inital model from keras folder
keras_model_copy0 = load_model('/home/somayeh/Desktop/Fei/result_last_model/best_model_CNN_MLP_result_last.h5', 
                               custom_objects={'my_metric':my_metric})
keras_model_copy0.compile(optimizer='adam', loss='mse', metrics=[my_metric])


# In[ ]:


# use iteration create variable names
createVar = locals()

# first eva. for test data 1; second eva. for test data 2;
first_evaluation = []
second_evaluation = []

iteration_times = 50 
# number of ANN
num_ann = 5 

acc = []
keras_model = []
# data to plot
times_stat = np.zeros([y_test1.shape[0], iteration_times]) 
times_roll = np.zeros([y_test1.shape[0], iteration_times])
mu1 = []
mu2 = []
std1 = []
std2 = []

# set data for ANN
batch_sizes = 32
epoch = 30

for iteration in range(iteration_times):

    iteration_folder = os.path.join(report_path, 'iteration_{}'.format(iteration))
    try:
        os.makedirs(iteration_folder)
    except:
        print('{} probably already exists'.format(iteration_folder))

    if iteration == 0:

        for number in range(1, num_ann):
            # copy model
            createVar['keras_model_copy'+str(number)] = clone_model(keras_model_copy0)
            createVar['keras_model_copy'+str(number)].set_weights(keras_model_copy0.get_weights())
            createVar['keras_model_copy'+str(number)].compile(
                                                            optimizer='adam',
                                                            loss='mse',
                                                            metrics=[my_metric]
                                                            )

            # create checkpoint
            createVar['checkpoint' + str(number)] = ModelCheckpoint(
                                        os.path.join(iteration_folder, 'model_' + str(number) + '_best.h5'),
                                        monitor='val_my_metric',
                                        verbose=1,
                                        mode='max',
                                        save_best_only=True)
            # train model
            createVar['history' + str(number)] = createVar['keras_model_copy'+str(number)].fit(
                                        x=[num_train_x, image_train],
                                        y=num_train_y,
                                        batch_size=batch_sizes,
                                        epochs=epoch,
                                        validation_split=0.04,
                                        callbacks=[createVar['checkpoint' + str(number)]]
                                        )

            # load best model from last training
            createVar['model_' + str(number) + '_best'] = load_model(
                                    os.path.join(iteration_folder, 'model_'+str(number)+'_best.h5'),
                                    custom_objects={'my_metric': my_metric})

            # append best model to a list, that will be needed in get_best_model function
            keras_model.append([createVar['model_' + str(number) + '_best']])

            # predict x_test1
            createVar['prediction' + str(number)] = createVar['model_' + str(number) + '_best'].predict([x_test1, image_test1])

            # calculate accuracy of each model
            acc.append(my_metric(y_test1, createVar['prediction' + str(number)]))

        # first model is exception, should be considered separately
        # insert 0 means insert that model at first location
        keras_model.append([keras_model_copy0])

        prediction0 = keras_model_copy0.predict([x_test1, image_test1])
        acc.append(my_metric(y_test1, prediction0))

        # get best model for one iteration
        best_model = get_best_model(acc, keras_model)
        best_model[0].save(os.path.join(iteration_folder, 'best_model_total.h5'))

        # load best and for 1st evaluation with test data 1 and step 2nd evaluation for test data 2

        best = load_model(os.path.join(iteration_folder, 'best_model_total.h5'),
                          custom_objects={'my_metric': my_metric})
        prediction_data1 = best.predict([x_test1, image_test1])
        prediction_data2 = best.predict([x_test2, image_test2])

        # data for plot times graph
        times = prediction_data1 / y_test1
        for m in range(times.shape[0]):
            for n in range(times.shape[1]):
                if times[m][n] <= 0:
                    times[m][n] = 0.01

        times_stat[:, iteration] = 10*np.log10(times[:, 0])
        times_roll[:, iteration] = 10*np.log10(times[:, 1])

        mu1.append(np.mean(10*np.log10(times[:, 0])))
        mu2.append(np.mean(10*np.log10(times[:, 1])))
        std1.append(np.std(10*np.log10(times[:, 0])))
        std2.append(np.std(10*np.log10(times[:, 1])))

        # write accuracy in file
        # with test data 2
        acc_eva1 = my_metric(y_test1, prediction_data1)
        acc_eva2 = my_metric(y_test2, prediction_data2)

        first_evaluation.append(acc_eva1)
        second_evaluation.append(acc_eva2)

        acc_stat_fric, acc_roll_fric = get_accuracy(y_test2, prediction_data2)

        with open(os.path.join(iteration_folder, 'accuracy.txt'), 'a+') as file:
            file.write(str(acc_stat_fric) + '\n')
            file.write(str(acc_roll_fric))

        # every time clear list for next iteration
        keras_model.clear()
        acc.clear()

    else:
        for number in range(1, num_ann):
            # model from last iteration best selected

            file_path = os.path.join(report_path,
                                     'iteration_{}'.format(iteration - 1),
                                     'best_model_total.h5'.format(iteration - 1)
                                     )
            model_best_last_circle = load_model(file_path, custom_objects={'my_metric': my_metric})

            # copy model
            createVar['keras_model_copy' + str(number)] = clone_model(model_best_last_circle)
            createVar['keras_model_copy' + str(number)].set_weights(keras_model_copy0.get_weights())
            createVar['keras_model_copy' + str(number)].compile(
                optimizer='adam',
                loss='mse',
                metrics=[my_metric]
            )

            # create checkpoint
            createVar['checkpoint' + str(number)] = ModelCheckpoint(
                os.path.join(iteration_folder, 'model_' + str(number) + '_best.h5'),
                monitor='val_my_metric',
                verbose=1,
                mode='max',
                save_best_only=True)
            # train model
            createVar['history' + str(number)] = createVar['keras_model_copy' + str(number)].fit(
                x=[num_train_x, image_train],
                y=num_train_y,
                batch_size=batch_sizes,
                epochs=epoch,
                validation_split=0.04,
                callbacks=[createVar['checkpoint' + str(number)]]
            )

            # load best model from last training
            createVar['model_' + str(number) + '_best'] = load_model(
                os.path.join(iteration_folder, 'model_' + str(number) + '_best.h5'),
                custom_objects={'my_metric': my_metric})

            # append best model to a list, that will be needed in get_best_model function
            keras_model.append([createVar['model_' + str(number) + '_best']])

            # predict x_test1
            createVar['prediction' + str(number)] = createVar['model_' + str(number) + '_best'].predict([x_test1, image_test1])

            # calculate accuracy of each model
            acc.append(my_metric(y_test1, createVar['prediction' + str(number)]))

        # first model is exception, should be considered separately
        # insert 0 means insert that model at first location
        keras_model.append([model_best_last_circle])

        prediction0 = model_best_last_circle.predict([x_test1, image_test1])
        acc.append(my_metric(y_test1, prediction0))

        # get best model for one iteration
        best_model = get_best_model(acc, keras_model)
        best_model[0].save(os.path.join(iteration_folder, 'best_model_total.h5'))

        # load best and for 1st evaluation with test data 1 and step 2nd evaluation for test data 2

        best = load_model(os.path.join(iteration_folder, 'best_model_total.h5'),
                          custom_objects={'my_metric': my_metric})
        prediction_data1 = best.predict([x_test1, image_test1])
        prediction_data2 = best.predict([x_test2, image_test2])

        times_stat[:, iteration] = 10*np.log10(times[:, 0])
        times_roll[:, iteration] = 10*np.log10(times[:, 1])

        mu1.append(np.mean(10*np.log10(times[:, 0])))
        mu2.append(np.mean(10*np.log10(times[:, 1])))
        std1.append(np.std(10*np.log10(times[:, 0])))
        std2.append(np.std(10*np.log10(times[:, 1])))

        # write accuracy in file
        # with test data 2
        acc_eva1 = my_metric(y_test1, prediction_data1)
        acc_eva2 = my_metric(y_test2, prediction_data2)

        first_evaluation.append(acc_eva1)
        second_evaluation.append(acc_eva2)

        acc_stat_fric, acc_roll_fric = get_accuracy(y_test2, prediction_data2)

        with open(os.path.join(iteration_folder, 'accuracy.txt'), 'a+') as file:
            file.write(str(acc_stat_fric) + '\n')
            file.write(str(acc_roll_fric))

        # clear list end this iteration
        keras_model.clear()
        acc.clear()


# In[ ]:


x = np.arange(0, iteration_times, 1)
plt.figure('acc')
plt.plot(x, first_evaluation)
plt.plot(x, second_evaluation)
plt.legend(['test_data_1', 'test_data_2'])
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.title('accuracy')
plt.savefig(os.path.join(report_path, 'accuracy'))
plt.show()


# In[ ]:


keras_model_copy0 = load_model('/home/somayeh/Desktop/Fei/report_RL/iteration_0/best_model_total.h5', 
                               custom_objects={'my_metric':my_metric})
keras_model_copy1 = load_model('/home/somayeh/Desktop/Fei/report_RL/iteration_10/best_model_total.h5', 
                               custom_objects={'my_metric':my_metric})
keras_model_best2 = load_model('/home/somayeh/Desktop/Fei/report_RL/iteration_25/best_model_total.h5', 
                               custom_objects={'my_metric':my_metric})
keras_model_best4 = load_model('/home/somayeh/Desktop/Fei/report_RL/iteration_49/best_model_total.h5', 
                               custom_objects={'my_metric':my_metric})


predict1=keras_model_copy0.predict([x_test1, image_test1])
predict2=keras_model_copy1.predict([x_test1, image_test1])
predict3=keras_model_best2.predict([x_test1, image_test1])
predict4=keras_model_best4.predict([x_test1, image_test1])


times1 = predict1 / y_test1
times1[times1<0]=0.01
times2 = predict2 / y_test1
times2[times2<0]=0.01
times3 = predict3 / y_test1
times3[times3<0]=0.01
times4 = predict4 / y_test1
times4[times4<0]=0.01

Static_Fric1 = times1[:,0]
Static_Fric1 = 10*np.log10(Static_Fric1)
mean1 = Static_Fric1.mean()
std1 = Static_Fric1.std()
num_bins = 100
n1, bins1, patches1 = plt.hist(Static_Fric1, num_bins, density=True)
y1 = norm.pdf(bins1, mean1, std1)
plt.plot(bins1, y1, label = 'iteration 1', color='y')

Static_Fric2 = times2[:,0]
Static_Fric2 = 10*np.log10(Static_Fric2)
mean2 = Static_Fric2.mean()
std2 = Static_Fric2.std()
n2, bins2, patches2 = plt.hist(Static_Fric2, num_bins, density=True)
y2 = norm.pdf(bins2, mean2, std2)
plt.plot(bins2, y2, label = 'iteration 10', color='c')

Static_Fric3 = times3[:,0]
Static_Fric3 = 10*np.log10(Static_Fric3)
mean3 = Static_Fric3.mean()
std3 = Static_Fric3.std()
num_bins = 100
n3, bins3, patches3 = plt.hist(Static_Fric3, num_bins, density=True)
y3 = norm.pdf(bins3, mean3, std3)
plt.plot(bins3, y3, label = 'iteration 25', color='k')


Static_Fric4 = times4[:,0]
Static_Fric4 = 10*np.log10(Static_Fric4)
mean4 = Static_Fric4.mean()
std4 = Static_Fric4.std()
n4, bins4, patches4 = plt.hist(Static_Fric4, num_bins, density=True)
y4 = norm.pdf(bins4, mean4, std4)
plt.plot(bins4, y4, label = 'iteration 50', color='b')

plt.xlabel('Deviation')
plt.ylabel('Probability')
plt.title('Deviation of predictions for static friction')
plt.legend()
plt.savefig('/home/somayeh/Desktop/Fei/report_RL/last_static.png')
plt.show()

Roll_Fric1 = times1[:,-1]
Roll_Fric1 = 10*np.log10(Roll_Fric1)
mean1 = Roll_Fric1.mean()
std1 = Roll_Fric1.std()
num_bins = 100
n1, bins1, patches1 = plt.hist(Roll_Fric1, num_bins, density=True)
y1 = norm.pdf(bins1, mean1, std1)
plt.plot(bins1, y1, label = 'iteration 1', color='y')

Roll_Fric2 = times2[:,-1]
Roll_Fric2 = 10*np.log10(Roll_Fric2)
mean2 = Roll_Fric2.mean()
std2 = Roll_Fric2.std()
num_bins = 100
n2, bins2, patches2 = plt.hist(Roll_Fric2, num_bins, density=True)
y2 = norm.pdf(bins2, mean2, std2)
plt.plot(bins2, y2, label = 'iteration 10', color='c')

Roll_Fric3 = times3[:,-1]
Roll_Fric3 = 10*np.log10(Roll_Fric3)
mean3 = Roll_Fric3.mean()
std3 = Roll_Fric3.std()
num_bins = 100
n3, bins3, patches3 = plt.hist(Roll_Fric3, num_bins, density=True)
y3 = norm.pdf(bins3, mean3, std3)
plt.plot(bins3, y3, label = 'iteration 25', color='r')

Roll_Fric4 = times4[:,-1]
Roll_Fric4 = 10*np.log10(Roll_Fric4)
mean4 = Roll_Fric4.mean()
std4 = Roll_Fric4.std()
num_bins = 100
n4, bins4, patches4 = plt.hist(Roll_Fric4, num_bins, density=True)
y4 = norm.pdf(bins4, mean4, std4)
plt.plot(bins4, y4, label = 'iteration 50', color='b')


plt.xlabel('Deviation')
plt.ylabel('Probability')
plt.title('Deviation of predictions for Rolling friction')
plt.legend()
plt.savefig('/home/somayeh/Desktop/Fei/report_RL/last_roll.png')
plt.show()


# In[ ]:


x_test1 = x_test1[:30]
image_test1 = image_test1[:30]
x_test2 = x_test2[:20]
image_test2 = image_test2[:20]
keras_model_copy0 = load_model('/home/somayeh/Desktop/Fei/report_RL/50 128 untrained/iteration_0/best_model_total.h5', 
                               custom_objects={'my_metric':my_metric})
keras_model_best = load_model('/home/somayeh/Desktop/Fei/report_RL/50 128 untrained/iteration_1/best_model_total.h5', 
                               custom_objects={'my_metric':my_metric})
prediction1 =keras_model_copy0.predict([x_test1, image_test1])
prediction2 =keras_model_copy0.predict([x_test2, image_test2])
prediction3 =keras_model_best.predict([x_test1, image_test1])
prediction4 =keras_model_best.predict([x_test2, image_test2])
y_test1=y_test1[:30]
y_test2=y_test2[:20]


s_t = 50
x_pt = np.arange(1,s_t+1)
x = np.arange(1,30+1)
x2 = x_pt[len(x):]
#s_t = len(x_test)
#x_pt = np.arange(1,s_t+1)
#x = np.arange(1,len(x_test)+1)
#x2 = x_pt[len(x):]

report_path = r'/home/somayeh/Desktop/Fei/report_RL'

fig = plt.figure(figsize=(16,12))

ax1 = fig.add_subplot(211)
lab1 = ax1.plot(x,y_test1[:,0], 'k',label='static_test')

ax1.plot(x,y_test1[:,0],'k*', x2,y_test2[:,0],'k', x2,y_test2[:,0],'k*')
lab2 = ax1.plot(x,prediction1[0:30,0], 'r',label='static_pred')

ax1.plot(x,prediction1[0:30,0],'r*', x2,prediction2[0:20,0],'r', x2,prediction2[0:20,0],'r*')

ax1.set_xlabel('test1 ---- initial model ---- test2',fontsize=15)
ax1.set_ylabel('Static Friction',fontsize=15)
ax1.set_xlim(0,len(x_pt)+1)

lab = lab1+lab2
labs = [j.get_label() for j in lab]
ax1.legend(lab, labs, loc='best',ncol=4,fontsize=12)

plt.axvspan(xmin=0,xmax=len(x)+0.5,facecolor='y',alpha=0.1)
plt.axvspan(xmin=len(x)+0.5,xmax=60,facecolor='c',alpha=0.1)              
               
plt.title('initial-best-comparision(test1,2 separately) of GRL',fontsize=16)
ax3 = fig.add_subplot(212)
ax3.plot(x,y_test1[:,0],'k',x,y_test1[:,0],'k*', x2,y_test2[:,0],'k', x2,y_test2[:,0],'k*')
ax3.plot(x,prediction3[0:30,0],'r',x,prediction3[0:30,0],'r*', x2,prediction4[0:20,0],'r', x2,prediction4[0:20,0],'r*')
ax3.set_xlabel('test1 ---- best model ---- test2',fontsize=15)
ax3.set_ylabel('Static Friction',fontsize=15)
ax3.set_xlim(0,len(x_pt)+1)
ax3.legend(lab, labs, loc='best',ncol=4,fontsize=12)
plt.axvspan(xmin=0,xmax=len(x)+0.5,facecolor='y',alpha=0.1)
plt.axvspan(xmin=len(x)+0.5,xmax=60,facecolor='c',alpha=0.1)
plt.savefig(os.path.join(report_path, 'partial-initial-best comparision(test1,2 separately) of GRL.png'))


# In[ ]:


fig = plt.figure(figsize=(16,12))

ax1 = fig.add_subplot(211)
lab1 = ax1.plot(x,y_test1[:,-1], 'k',label='rolling_test')

ax1.plot(x,y_test1[:,-1],'k*', x2,y_test2[:,-1],'k', x2,y_test2[:,-1],'k*')
lab2 = ax1.plot(x,prediction1[0:30,-1], 'r',label='rolling_pred')

ax1.plot(x,prediction1[0:30,-1],'r*', x2,prediction2[0:20,-1],'r', x2,prediction2[0:20,-1],'r*')

ax1.set_xlabel('test1 ---- initial model ---- test2',fontsize=15)
ax1.set_ylabel('Rolling Friction',fontsize=15)
ax1.set_xlim(0,len(x_pt)+1)

lab = lab1+lab2
labs = [j.get_label() for j in lab]
ax1.legend(lab, labs, loc='best',ncol=4,fontsize=12)

plt.axvspan(xmin=0,xmax=len(x)+0.5,facecolor='y',alpha=0.1)
plt.axvspan(xmin=len(x)+0.5,xmax=60,facecolor='c',alpha=0.1)              
               
plt.title('initial-best-comparision(test1,2 separately) of GRL',fontsize=16)
ax3 = fig.add_subplot(212)
ax3.plot(x,y_test1[:,-1],'k',x,y_test1[:,-1],'k*', x2,y_test2[:,-1],'k', x2,y_test2[:,-1],'k*')
ax3.plot(x,prediction3[0:30,-1],'r',x,prediction3[0:30,-1],'r*', x2,prediction4[0:20,-1],'r', x2,prediction4[0:20,-1],'r*')
ax3.set_xlabel('test1 ---- best model ---- test2',fontsize=15)
ax3.set_ylabel('Rolling Friction',fontsize=15)
ax3.set_xlim(0,len(x_pt)+1)
ax3.legend(lab, labs, loc='best',ncol=4,fontsize=12)
plt.axvspan(xmin=0,xmax=len(x)+0.5,facecolor='y',alpha=0.1)
plt.axvspan(xmin=len(x)+0.5,xmax=60,facecolor='c',alpha=0.1)
plt.savefig(os.path.join(report_path, 'partial-initial-best-comparision(test1,2 separately) rolling.png'))


# In[ ]:


# select 'best model' after iteration to draw scatter distribution!
# scatter distribution before log
prediction3 =keras_model_best.predict([x_test1, image_test1])  # select best model
times = predict3 / y_test1
times[times<0]=0.01
plt.scatter(times[:, 0], times[:, -1], alpha=0.4)
plt.xlabel('Static Friction')
plt.ylabel('Rolling Friction')
plt.axhline(y=0,ls="-",c="red")
plt.axvline(x=0,ls="-",c="red")
plt.axhline(y=1,ls="-",c="g")
plt.axvline(x=1,ls="-",c="g")
plt.text(0,0,(0,0),color='r')
plt.text(1,1,(1,1),color='g')
plt.title('Scattered distribution for Friction')
plt.savefig(os.path.join(report_path, 'Scattered distribution'))
plt.show()

# scatter distribution after log
times_stat = 10*np.log10(times[:, 0])
times_roll = 10*np.log10(times[:, -1])
plt.scatter(times_stat, times_roll, alpha=0.4)
plt.xlabel('Static Friction')
plt.ylabel('Rolling Friction')
plt.axhline(y=0,ls="-",c="red")
plt.axvline(x=0,ls="-",c="red")
plt.text(0,0,(0,0),color='r')
plt.title('Scattered distribution for Friction')
plt.savefig(os.path.join(report_path, 'Scattered distribution for Friction all'))
plt.show()

