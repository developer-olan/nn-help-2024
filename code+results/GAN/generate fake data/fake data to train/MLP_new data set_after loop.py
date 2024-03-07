#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from keras.models import clone_model
from keras.callbacks import ModelCheckpoint
from scipy.interpolate import make_interp_spline
from keras import backend as K
from scipy.stats import norm


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


# dataset preparation
# report path setting
report_path = r'/home/hashemi/Fei Shao/fake data/MLP results'

# read data
df = pd.read_excel('/home/hashemi/Fei Shao/total_fake.xlsx', dtype=np.float32)
df = df.drop(df[df.iloc[:,7]>=0.1].index) 

# spilt train and test data
y = df[['Static_Fric', 'Rolling_Fric']].values
x = df.drop(labels=['Static_Fric', 'Rolling_Fric'], axis=1)

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


# spilt test1 and test2
x_test1, x_test2, y_test1, y_test2 = train_test_split(
    x_test, y_test,
    test_size=0.4,
    shuffle=True
)

# load model from keras best selected
# copy0 means the inital model from keras folder
keras_model_copy0 = load_model(r'/home/hashemi/Fei Shao/fake data/MLP results/best_model_delete.h5',
                              custom_objects={'my_metric':my_metric})
# keras_model_inital.summary()

keras_model_copy0.compile(
    optimizer='adam',
    loss='mse',
    metrics=[my_metric]
)


# In[ ]:


# use iteration create variable names
createVar = locals()

# first eva. for test data 1; second eva. for test data 2;
# times mean how many times bigger or smaller
first_evaluation = []
second_evaluation = []

iteration_times = 10
# number of ANN
num_ann = 10

acc = []
keras_model = []
# data to plot
times_stat = np.zeros([y_test1.shape[0], iteration_times])  # y-test1 shape 831; iteration 10 time
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
                                        x=x_train,
                                        y=y_train,
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
            createVar['prediction' + str(number)] = createVar['model_' + str(number) + '_best'].predict(x_test1)

            # calculate accuracy of each model
            acc.append(my_metric(y_test1, createVar['prediction' + str(number)]))

        # first model is exception, should be considered separately
        # insert 0 means insert that model at first location
        keras_model.append([keras_model_copy0])

        prediction0 = keras_model_copy0.predict(x_test1)
        acc.append(my_metric(y_test1, prediction0))

        # get best model for one iteration
        best_model = get_best_model(acc, keras_model)
        best_model[0].save(os.path.join(iteration_folder, 'best_model_total.h5'))

        # load best and for 1st evaluation with test data 1 and step 2nd evaluation for test data 2

        best = load_model(os.path.join(iteration_folder, 'best_model_total.h5'),
                          custom_objects={'my_metric': my_metric})
        prediction_data1 = best.predict(x_test1)
        prediction_data2 = best.predict(x_test2)

        # data for plot times graph
        times = prediction_data1 / y_test1
        for m in range(times.shape[0]):
            for n in range(times.shape[1]):
                if times[m][n] <= 0:
                    times[m][n] = 0.01

        times_stat[:, iteration] = 10 * np.log10(times[:, 0])
        times_roll[:, iteration] = 10 * np.log10(times[:, 1])

        mu1.append(np.mean(10 * np.log10(times[:, 0])))
        mu2.append(np.mean(10 * np.log10(times[:, 1])))
        std1.append(np.std(10 * np.log10(times[:, 0])))
        std2.append(np.std(10 * np.log10(times[:, 1])))

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
            createVar['keras_model_copy' + str(number)].set_weights(model_best_last_circle.get_weights())
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
                x=x_train,
                y=y_train,
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
            createVar['prediction' + str(number)] = createVar['model_' + str(number) + '_best'].predict(x_test1)

            # calculate accuracy of each model
            acc.append(my_metric(y_test1, createVar['prediction' + str(number)]))

        # first model is exception, should be considered separately
        # insert 0 means insert that model at first location
        keras_model.append([model_best_last_circle])

        prediction0 = model_best_last_circle.predict(x_test1)
        acc.append(my_metric(y_test1, prediction0))

        # get best model for one iteration
        best_model = get_best_model(acc, keras_model)
        best_model[0].save(os.path.join(iteration_folder, 'best_model_total.h5'))

        # load best and for 1st evaluation with test data 1 and step 2nd evaluation for test data 2

        best = load_model(os.path.join(iteration_folder, 'best_model_total.h5'),
                          custom_objects={'my_metric': my_metric})
        prediction_data1 = best.predict(x_test1)
        prediction_data2 = best.predict(x_test2)
        times = prediction_data1 / y_test1
        for m in range(times.shape[0]):
            for n in range(times.shape[1]):
                if times[m][n] <= 0:
                    times[m][n] = 0.01

        times_stat[:, iteration] = 10 * np.log10(times[:, 0])
        times_roll[:, iteration] = 10 * np.log10(times[:, 1])

        mu1.append(np.mean(10 * np.log10(times[:, 0])))
        mu2.append(np.mean(10 * np.log10(times[:, 1])))
        std1.append(np.std(10 * np.log10(times[:, 0])))
        std2.append(np.std(10 * np.log10(times[:, 1])))

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
plt.savefig(os.path.join(report_path, 'accuracy.png'))
plt.show()

# plot bigger / smaller
num_bins = 100
plt.figure('stat_fri')
for j in range(len(mu1)):
    n1, bins1, patches1 = plt.hist(times_stat[:, j], num_bins, density=True)
    y1 = norm.pdf(bins1, mu1[j], std1[j])
    if j == 0 or j == int(len(mu1)/3) or j == int(len(mu1)/2) or j == int(2*len(mu1)/3) or j == len(mu1)-1:
        plt.plot(bins1, y1, label='iteration'+str(j+1))
plt.xlabel('Times')
plt.ylabel('Probability')
plt.title('Times-Probability History Graph Static Friction')
plt.legend()
plt.savefig(os.path.join(report_path, 'stat_fri'))

plt.figure('roll_fri')
for k in range(len(mu2)):
    n2, bins2, patches2 = plt.hist(times_roll[:, k], num_bins, density=True)
    y2 = norm.pdf(bins2, mu2[k], std2[k])
    if k == 0 or k == int(len(mu2)/3) or k == int(len(mu2)/2) or k == int(2*len(mu2)/3) or k == len(mu2)-1:
        plt.plot(bins2, y2, label='iteration'+str(k+1))
plt.xlabel('Times')
plt.ylabel('Probability')
plt.title('Times-Probability History Graph Roll Friction')
plt.legend()
plt.savefig(os.path.join(report_path, 'roll_fri'))


# In[ ]:


# select best model manually to draw
plt.scatter(times[:, 0], times[:, 1], alpha=0.4)
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

plt.scatter(times_stat[:, iteration], times_roll[:, iteration], alpha=0.4)
plt.xlabel('Static Friction')
plt.ylabel('Rolling Friction')
plt.axhline(y=0,ls="-",c="red")
plt.axvline(x=0,ls="-",c="red")
plt.text(0,0,(0,0),color='r')
plt.title('Scattered distribution for Friction')
plt.savefig(os.path.join(report_path, 'Scattered distribution for Friction all'))
plt.show()

plt.scatter(times_stat[:, iteration], times_roll[:, iteration], alpha=0.4)
plt.xlabel('Static Friction')
plt.ylabel('Rolling Friction')
plt.axhline(y=0,ls="-",c="red")
plt.axvline(x=0,ls="-",c="red")
plt.text(0,0,(0,0),color='r')
plt.xlim(-5,5)
plt.ylim(-10,10)
plt.title('Scattered distribution for Friction')
plt.savefig(os.path.join(report_path, 'Scattered distribution for Friction'))
plt.show()

