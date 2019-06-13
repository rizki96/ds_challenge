# https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/

import numpy as np
from math import sqrt
from numpy import array
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed

# evaluate one or more weekly forecasts against expected values
#def evaluate_forecasts(actual, predicted):
#	scores = list()
#	# calculate an RMSE score for each steps
#	for i in range(actual.shape[1]):
#		# calculate mse
#		mse = mean_squared_error(actual[:, i], predicted[:, i])
#		# calculate rmse
#		rmse = sqrt(mse)
#		# store
#		scores.append(rmse)
#	# calculate overall RMSE
#	s = 0
#	for row in range(actual.shape[0]):
#		for col in range(actual.shape[1]):
#			s += (actual[row, col] - predicted[row, col])**2
#	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
#	return score, scores

# summarize scores
#def summarize_scores(name, score, scores):
#	s_scores = ', '.join(['%.1f' % s for s in scores])
#	print('%s: [%.3f] %s' % (name, score, s_scores))

# convert history into inputs and outputs
#def to_supervised(train, n_input, n_out=7):
#	# flatten data
#	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
#	X, y = list(), list()
#	in_start = 0
#	# step over the entire history one time step at a time
#	for _ in range(len(data)):
#		# define the end of the input sequence
#		in_end = in_start + n_input
#		out_end = in_end + n_out
#		# ensure we have enough data for this instance
#		if out_end < len(data):
#			x_input = data[in_start:in_end, 0]
#			x_input = x_input.reshape((len(x_input), 1))
#			X.append(x_input)
#			y.append(data[in_end:out_end, 0])
#		# move along one time step
#		in_start += 1
#	return array(X), array(y)

# train the model
#def build_model(train, n_input):
#	# prepare data
#	train_x, train_y = to_supervised(train, n_input)
#	# define parameters
#	verbose, epochs, batch_size = 0, 70, 16
#	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
#	# define model
#	model = Sequential()
#	model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
#	model.add(Dense(100, activation='relu'))
#	model.add(Dense(n_outputs))
#	model.compile(loss='mse', optimizer='adam')
#	# fit network
#	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
#	return model

# make a forecast
#def forecast(model, history, n_input):
#	# flatten data
#	data = array(history)
#	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
#	# retrieve last observations for input data
#	input_x = data[-n_input:, 0]
#	# reshape into [1, n_input, 1]
#	input_x = input_x.reshape((1, len(input_x), 1))
#	# forecast the next week
#	yhat = model.predict(input_x, verbose=0)
#	# we only want the vector forecast
#	yhat = yhat[0]
#	return yhat

# evaluate a single model
#def evaluate_model(train, test, n_input):
#	# fit model
#	model = build_model(train, n_input)
#	# history is a list of daily data
#	history = [x for x in train]
#	# walk-forward validation over each week
#	predictions = list()
#	for i in range(len(test)):
#		# predict the week
#		yhat_sequence = forecast(model, history, n_input)
#		# store the predictions
#		predictions.append(yhat_sequence)
#		# get real observation and add to history for predicting the next day
#		history.append(test[i, :])
#	# evaluate predictions every 15 minutes for each day
#	predictions = array(predictions)
#	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
#	return score, scores

###### another implementation ######

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
    model = Sequential()
    model.add(LSTM(batch_input_shape=(train_batch_size, None, dim_in),
                   return_sequences=True, units=nb_units, stateful=True))
    model.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
    model.compile(loss = 'mse', optimizer = 'rmsprop')
    return model

def train_model(model, inputs, outputs, inputs_test, outputs_test,
                N, T, epochs, train_batch_size, test_batch_size, T_after_cut):
    nb_reset = int(N / train_batch_size)
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

#def predict(model, path, n=0, interval=1):
#    load_model(path)
#    # After 100 epochs: loss: 0.0048 / val_loss: 0.0047. 
#
#    idx = range(n, n+interval)
#    x = inputs_test[idx].flatten()
#    y_hat = model.predict(inputs_test[idx]).flatten()
#    y = outputs_test[idx].flatten()
#    return x, y, yhat
