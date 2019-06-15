import numpy as np
from math import sqrt
from math import ceil
from numpy import array
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import Callback

def stateful_cut(arr, batch_size, T_after_cut):
    if len(arr.shape) != 3:
        # N: Independent sample size,
        # T: Time length,
        # m: Dimension
        print("ERROR: please format arr as a (N, T, m) array.")
        # if len(arr.shape) == 1, reshape is as follows:
        # N = 1
        # T = arr.shape[0]
        # m = 1
        # arr.reshape(N, T, m)
        #
        # if len(arr.shape) == 2, there are two ways to reshape:
        # N = arr.shape[0]
        # T = arr.shape[1]
        # m = 1
        # arr.reshape(N, T, m)
        #     or
        # N = 1
        # T = arr.shape[0]
        # m = arr.shape[1]
        # arr.reshape(N, T, m)
    N = arr.shape[0]
    T = arr.shape[1]
    
    # We need T_after_cut * nb_cuts = T
    nb_cuts = int(T / T_after_cut)
    if nb_cuts * T_after_cut != T:
        print("ERROR: T_after_cut must divide T")
    
    # We need batch_size * nb_reset = N
    # If nb_reset = 1, we only reset after the whole epoch
    nb_reset = int(N / batch_size)
    if nb_reset * batch_size != N:
        print("ERROR: batch_size must divide N")

    # We can observe:
    # nb_batches = (N*T)/(T_after_cut*batch_size)
    # nb_batches = nb_reset * nb_cuts

    # Cutting (technical)
    cut1 = np.split(arr, nb_reset, axis=0)
    cut2 = [np.split(x, nb_cuts, axis=1) for x in cut1]
    cut3 = [np.concatenate(x) for x in cut2]
    cut4 = np.concatenate(cut3)
    return(cut4)

def define_reset_states_class(nb_cuts):
    class ResetStatesCallback(Callback):
        def __init__(self):
            self.counter = 0
    
        def on_batch_begin(self, batch, logs={}):
            # We reset states when nb_cuts batches are completed, as
            # shown in the after cut figure
            if self.counter % nb_cuts == 0:
                self.model.reset_states()
            self.counter += 1
            
        def on_epoch_end(self, epoch, logs={}):
            # reset states after each epoch
            self.model.reset_states()
    return(ResetStatesCallback)

def batched(i, arr, batch_size):
    return(arr[i*batch_size:(i+1)*batch_size])

def test_on_batch_stateful(model, inputs, outputs, batch_size, nb_cuts):
    nb_batches = int(len(inputs)/batch_size)
    sum_pred = 0
    for i in range(nb_batches):
        if i % nb_cuts == 0:
            model.reset_states()
        x = batched(i, inputs, batch_size)
        y = batched(i, outputs, batch_size)
        sum_pred += model.test_on_batch(x, y)
    mean_pred = sum_pred / nb_batches
    return(mean_pred)

def define_stateful_val_loss_class(inputs, outputs, batch_size, nb_cuts):
    class ValidationCallback(Callback):
        def __init__(self):
            self.val_loss = []
    
        def on_epoch_end(self, epoch, logs={}):
            mean_pred = test_on_batch_stateful(self.model, inputs, outputs, 
                                               batch_size, nb_cuts)
            print('val_loss: {:0.3e}'.format(mean_pred), end = '')
            self.val_loss += [mean_pred]
            
        def get_val_loss(self):
            return(self.val_loss)
            
    return(ValidationCallback)

def build_model(train_batch_size, nb_units, dim_in, dim_out, T_after_cut):
    train_batch_size = train_batch_size if train_batch_size < 32 else (32*ceil(train_batch_size / 32))

    model = Sequential()
    model.add(LSTM(batch_input_shape=(train_batch_size, None, dim_in),
                   return_sequences=True, units=nb_units, stateful=True))
    model.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
    model.compile(loss = 'mse', optimizer = 'rmsprop')
    return model

def train_model(model, inputs, outputs, inputs_test, outputs_test,
                N, T, epochs, train_batch_size, test_batch_size, T_after_cut):
    # limit the batch size
    train_batch_size = train_batch_size if train_batch_size < 32 else (32*ceil(train_batch_size / 32))
    test_batch_size = test_batch_size if test_batch_size < 32 else (32*ceil(test_batch_size / 32))

    nb_reset = int(ceil(N / train_batch_size))
    nb_cuts = int(T / T_after_cut)
    if nb_reset > 1:
        ResetStatesCallback = define_reset_states_class(nb_cuts)
        ValidationCallback = define_stateful_val_loss_class(inputs_test,
                                                            outputs_test,
                                                            int(test_batch_size), nb_cuts)
        validation = ValidationCallback()
        history = model.fit(inputs, outputs, epochs = epochs,
                            batch_size = train_batch_size, shuffle=False,
                            callbacks = [ResetStatesCallback(), validation])
        history.history['val_loss'] = ValidationCallback.get_val_loss(validation)
    else:
        # If nb_reset = 1, we should reset states after each epoch.
        # To improve computational speed, we can decide not to reinitialize states
        # at all. Results are similar in this case.
        # In the following line, states are not reinitialized at all:
        history = model.fit(inputs, outputs, epochs = epochs,
                            batch_size = train_batch_size, shuffle=False,
                            validation_data=(inputs_test, outputs_test))
    return history

def plotting(history):
    plt.plot(history.history['loss'], color = "red")
    plt.plot(history.history['val_loss'], color = "blue")
    red_patch = mpatches.Patch(color='red', label='Training')
    blue_patch = mpatches.Patch(color='blue', label='Test')
    plt.legend(handles=[red_patch, blue_patch])
    plt.xlabel('Epochs')
    plt.ylabel('MSE loss')
    plt.show()
