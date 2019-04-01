from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils import np_utils
from keras.datasets import cifar10
from kerassurgeon import Surgeon
from kerassurgeon import Surgeon
from math import ceil
import keras
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
from keras.models import load_model
import cifar10vgg
import time


import logging
logger = logging.getLogger('pruner')
logger.setLevel(logging.DEBUG)# create file handler which logs even debug messages

current_stamp = time.asctime()

fh = logging.FileHandler('prune_{}.log'.format(current_stamp))
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

vls = 'State {} : Validation Accuracy {} : Validation Loss {}'
tls = 'State {} : Test Accuracy {} : Test Loss {}'

img_width, img_height = 32, 32
train_data_dir = "./data/train"
nb_train_samples = 40125
nb_validation_samples = 4066
'''conv_layers = ['block1_conv1', 'block1_conv2', 'block2_conv1',
               'block2_conv2', 'block3_conv1', 'block3_conv2',
               'block3_conv3', 'block3_conv4', 'block4_conv1',
               'block4_conv2', 'block4_conv3', 'block4_conv4',
               'block5_conv1', 'block5_conv2', 'block5_conv3',
               'block5_conv4']  # list of all conv layers in the VGG19 model
'''
conv_layers = ['conv2d_'+str(i) for i in range(1,14)]
conv_layers.reverse()

def normalize(x):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)


datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    preprocessing_function=normalize,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2,
    horizontal_flip=True)

def top_3_accuracy(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


def train(model, epochs):
    bs = 1024
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=0.0001),
                  metrics=[top_3_accuracy,keras.metrics.categorical_crossentropy])
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=bs, subset='training'),
                        steps_per_epoch=len(x_train) / bs, epochs=epochs, verbose=1)
    return model

def compile_model(model):
    bs = 1024
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=0.0001),
                  metrics=[top_3_accuracy,keras.metrics.categorical_crossentropy])
    return model


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def takeSecond(elem):
    return elem[1]


if __name__ == "__main__":
    params = []
    accuracy_list = []
    test_accuracy_list=[]

    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train[:45000, :, :, :]
    y_train = y_train[:45000, :]

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # load the VGG16 model
    #model = applications.VGG19(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))
    model = cifar10vgg.cifar10vgg(False)
    model = model.model

    # freeze VGG layers
    for layer in model.layers:
        layer.trainable = True
    last = model.output
    #logger.info(last)
    #x = Flatten()(last)
    # x = Dense(1024, activation="relu")(x)
    # x = Dropout(0.5)(x)
    # x = Dense(1024, activation="relu")(x)
    #predictions = Dense(10, activation="softmax")(x)
    model = Model(input=model.input, output=model.output)

    logger.info("Starting initial training of the model")

    if os.path.isfile('./my_model_initial.h5'):
        logger.info('Model found on disk. Loading..')
        model = load_model('my_model_initial.h5')
        model.summary()
    else:
        logger.info('Model not found on disk.. fine tuning..')
        model = compile_model(model)
        val_acc, val_loss = model.evaluate_generator(datagen.flow(x_train, y_train, subset='validation'), steps=len(x_train) / 1024, verbose=0)[1:]
        test_acc, test_loss = model.evaluate(x_test, y_test, batch_size=256)[1:]

        logger.info(vls.format("ORIGINAL" , val_acc , val_loss))
        logger.info(tls.format("ORIGINAL" , test_acc , test_loss))
        #logger.info('ORIGINAL top-3 validation accuracy of the model is :: ', val_acc)
        #logger.info('ORIGINAL top-3 test accuracy of the model is :: ', test_acc)

        for i in range(2):
                model = train(model, 5)
                val_acc, val_loss = model.evaluate_generator(datagen.flow(x_train , y_train, subset='validation'), steps=len(x_train) / 1024, verbose=0)[1:]
                test_acc, test_loss = model.evaluate(x_test, y_test, batch_size= 256)[1:]

                logger.info(vls.format("Finetuning {}".format(i), val_acc, val_loss))
                logger.info(tls.format("Finetuning {}".format(i), test_acc, test_loss))

        logger.info('Finished initial finetuning \n')
        model.save('my_model_initial.h5')
        logger.info('Cached the finetuned model \n')

    raise Exception("End")
    most_compressed_model_so_far_name = 'my_model_initial.h5'

    model = compile_model(model)

    # find the initial validation accuracy of the model before pruning
    val_acc,val_loss = model.evaluate_generator(datagen.flow(x_train, y_train, subset='validation'), steps=len(x_train) / 1024, verbose=0)[1:]
    test_acc,test_loss = model.evaluate(x_test, y_test, batch_size=256)[1:]
    
    logger.info(vls.format('INITIAL'), val_acc, val_loss)
    logger.info(tls.format('INITIAL'), tal_acc, tal_loss)

    params.append(1)
    accuracy_list.append(val_acc)
    test_acc, test_loss = model.evaluate(x_test, y_test, batch_size=256)[1:]
    test_accuracy_list.append(test_acc)
    val_accuracy_list = []
    
    acc_pruned = val_acc
    # find the initial number of params before pruning
    initial_params = model.count_params()
    conv_index = 0  # index of which conv_layer the surgeon is working on. Start from 0th conv layer
    default_proportion = 0.5
    proportion = default_proportion

    #Do until we reach the last layer.
    while conv_index<len(conv_layers):

        logger.info("\t\t ------ SURGERY BEGIN ------- \n")

        logger.info('Operating on layer :: ', conv_layers[conv_index])
        W = model.get_layer(conv_layers[conv_index]).get_weights()[0]
        ratio_list = []
        for i in range(W.shape[3]):
            l2_norm = np.linalg.norm(W[:,:,:,i])
            ratio_list.append((i, l2_norm))  # append norm to ratio list along with channel number

        ratio_list = sorted(ratio_list, key=takeSecond, reverse=True)
        logger.info(len(ratio_list))

        surgeon = Surgeon(model)
        number_of_channels_to_prune_at_once = int(len(ratio_list) * proportion)  # remove 50% of the channels at once

        
        #Go ahead with the process only if the number of channels to prune is greater than 1.
        if(number_of_channels_to_prune_at_once<=2):
            logger.info('pruned all filters in layer :: ' + str(conv_index) + '. Moving to the next layer to the left')
            conv_index = conv_index + 1
        else:
            channels_to_prune = []
            for z in range(number_of_channels_to_prune_at_once):
                channels_to_prune.append(ratio_list[z][0])
            surgeon.add_job('delete_channels', model.get_layer(conv_layers[conv_index]), channels=channels_to_prune)
            model = surgeon.operate()
            logger.info('surgery finished. % of parameters left now :: ', model.count_params() / initial_params)
            # train for 1 epochs
            logger.info("\t\t ------ SURGERY END ------- \n\n")

            retrain_iters = 5
            logger.info("Recovering from the pruned model by retraining for {} iterations...\n".format(retrain_iters))
            model = train(model, 5)
            logger.info("Recoverd by training \n")

            # find validation accuracy of the pruned model
            val_acc_pruned,val_loss_pruned = \
            model.evaluate_generator(datagen.flow(x_train, y_train, subset='validation'), steps=len(x_train) / 1024,
                                     verbose=0)[1:]
            threshold_for_pruning = 0.10
            if val_acc - val_acc_pruned<threshold_for_pruning:
                test_acc_pruned , test_loss_pruned = model.evaluate(x_test, y_test, batch_size=256)[1:]

                #Logger.Infoing status
                logger.info(vls.format("POST RECOVERY"), val_acc_pruned, val_loss_pruned)
                logger.info(tls.format("POST RECOVERY"), test_acc_pruned, test_loss_pruned)

                logger.info('validation accuracy has dropped by :: %f \n' % (val_acc - val_acc_pruned))

                #Bookkeeping for future plots.
                accuracy_list.append(val_acc_pruned)
                pruned_fraction = model.count_params() / initial_params
                params.append(pruned_fraction)
                test_accuracy_list.append(test_acc)
                np.savetxt("params.csv", params, delimiter=",")
                np.savetxt("accuracy_list.csv", accuracy_list, delimiter=",")
                np.savetxt("test_accuracy_list.csv", test_accuracy_list, delimiter=",")

                #Saving model for best case use.
                most_compressed_model_so_far_name = 'model_latest_pruned_{}_{}.h5'.format(pruned_fraction,val_acc_pruned)
                model.save(most_compressed_model_so_far_name)
                logger.info('Saved the model ... to {} \n'.format(most_compressed_model_so_far_name))
                val_acc = val_acc_pruned
                val_loss = val_loss_pruned

            else:
                logger.info('Not pruning the current layer ! Accuracy will drop ! \n')

                #restore the previous best model
                model = load_model(most_compressed_model_so_far_name)  


    accuracy_list = np.asarray(accuracy_list)
    test_accuracy_list = np.asarray(test_accuracy_list)

    '''fig, ax = plt.subplots()
    ax.plot(params, accuracy_list,'ro')
    ax.set(xlabel='% of parameters', ylabel='validation accuracy',
           title='L2 norm pruning')
    ax.grid()
    fig.savefig("test1.png")
    plt.show()
    params = np.asarray(params)
    '''


