# Python version: 3.7.7
# Tensorflow-gpu version: 1.14.0
# Keras version: 2.2.4-tf


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
"""

                            Attention - no teacher forcing 
                            
             These are not proper implementations of attention mechanism
                                
"""
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------


# ----
# Simulate some toy data 1

import numpy as np
from random import randint
from numpy import array
from numpy import argmax

# Data description: 
# Input is a sequence of n_in numbers. Target is the first n_out elements of the input sequence 

# generate a sequence of random integers
def gen_sequence(length, n_unique): 
	return [randint(1, n_unique-1) for _ in range(length)]

def gen_in_out(n_in, n_out, n_unique):
	# generate random sequence
    X = gen_sequence(n_in, n_unique)
    y = X[:n_out]
    X = np.array(X)
    y = np.array(y)
    return X,y

# generate ONE random sequence 
k_features = 60 #  
n_steps_in = 10 # time steps in 
n_steps_out = 4 # time steps out

X, y = gen_in_out(n_steps_in, n_steps_out, k_features)
print('X=%s, y=%s' % (X[0], y[0]))


# generate WHOLE training dataset of sample size=100,000
def gen_data(n_steps_in, n_steps_out, k_features, n_samples):
    X, y = list(), list()
    for _ in range(n_samples):
        X_tmp, y_tmp = gen_in_out(n_steps_in, n_steps_out, k_features)
        # store (create all inputs)
        X.append(X_tmp)
        y.append(y_tmp)
    return array(X), array(y)

X, y = gen_data(n_steps_in, n_steps_out, k_features, 100000)
  
# =============================================================================
# Bahdanau Attention layer


import tensorflow as tf

class BahdanauAttention(tf.keras.Model):
   # notice this is Model (below it is layer.Layer) but it does not matter
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)
      
    # decoder's s_t (previous hidden) or QUERY = dec_hidden (in examples below you will be sending encoders: state_c or state_h)
    # encoder outputs (hiden and output) or VALUES = enc_output (in examples below you will be sending encoders outputs)

  def call(self, dec_hidden, enc_output): 

    score = self.V(tf.nn.tanh(
        self.W1(dec_hidden_with_time_axis) + self.W2(enc_output)))

    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * enc_output

    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

# -----
# LSTM

import tensorflow.keras as keras
from tensorflow.keras.layers import Concatenate, Dense, Input, LSTM, Embedding, RepeatVector, TimeDistributed

embed_size=10
rnn_cell_size = 128

sequence_input = Input(shape=(n_steps_in, ))
embedded_sequences = Embedding(k_features, embed_size, input_length=n_steps_in)(sequence_input)
lstm_1 = LSTM(rnn_cell_size, return_sequences = True)(embedded_sequences)
outputs, state_h, state_c  = LSTM(rnn_cell_size, return_sequences=True, return_state=True)(lstm_1)
# "outputs" - hidden state which goes to the next cell and also is an output at each time step
# "cell state" - long term state transferred only to the next LSTM cell 
# "hidden state" - final hidden state 
context_vector, attention_weights = BahdanauAttention(10)(state_c, outputs) 

repeat = RepeatVector(n_steps_out)(context_vector)
lstm_2 = LSTM(rnn_cell_size, return_sequences = True)(repeat)
output = TimeDistributed(Dense(k_features, activation='softmax')) (lstm_2)

model = keras.models.Model(inputs=[sequence_input], outputs=[output])
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, epochs=5, validation_split=0.2, verbose=1, batch_size=100)

# Making predictions using model
for _ in range(10):
    X_test, y_test = gen_data(n_steps_in, n_steps_out, k_features, 1)
    probs = model.predict(X_test, verbose=0) # softmax probabilities 
    y_hat = np.argmax(probs, axis=-1)  # turn probabilities to values 
    print('X=%s y=%s, yhat=%s' % (X_test[0], y_test[0], y_hat[0]))
    
# -----
# LSTM Bidirectional 

import tensorflow.keras as keras
from tensorflow.keras.layers import Bidirectional

embed_size=10
rnn_cell_size = 128

sequence_input = Input(shape=(n_steps_in, ))
embedded_sequences = Embedding(k_features, embed_size, input_length=n_steps_in)(sequence_input)
lstm_1 = Bidirectional(LSTM(rnn_cell_size, return_sequences = True))(embedded_sequences)
outputs, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(rnn_cell_size, return_sequences=True, return_state=True))(lstm_1)

# concatenate the hidden states from each RNN - encoder state
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
context_vector, attention_weights = BahdanauAttention(10)(state_c, outputs)
repeat = RepeatVector(n_steps_out)(context_vector)
lstm_2 = Bidirectional(LSTM(rnn_cell_size, return_sequences = True))(repeat)
output = TimeDistributed(Dense(k_features, activation='softmax')) (lstm_2)

model = keras.models.Model(inputs=[sequence_input], outputs=[output]) 
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, y, epochs=5, validation_split=0.2, verbose=1, batch_size=100)

# Making predictions using model
for _ in range(10):
    X_test, y_test = gen_data(n_steps_in, n_steps_out, k_features, 1)
    probs = model.predict(X_test, verbose=0) # softmax probabilities 
    y_hat = np.argmax(probs, axis=-1)  # turn probabilities to values 
    print('X=%s y=%s, yhat=%s' % (X_test[0], y_test[0], y_hat[0]))
    
# -----
# GRU

from tensorflow.keras.layers import GRU

embed_size=10
rnn_cell_size = 128

sequence_input = Input(shape=(n_steps_in, ))
embedded_sequences = Embedding(k_features, embed_size, input_length=n_steps_in)(sequence_input)
gru_1 = GRU(rnn_cell_size, return_sequences = True)(embedded_sequences)
outputs, state_h  = GRU(rnn_cell_size, return_sequences=True, return_state=True)(gru_1)
context_vector, attention_weights = BahdanauAttention(10)(state_h, outputs)
repeat = RepeatVector(n_steps_out)(context_vector)
gru_2 = GRU(rnn_cell_size, return_sequences = True)(repeat)
output = TimeDistributed(Dense(k_features, activation='softmax')) (gru_2)

model = keras.models.Model(inputs=[sequence_input], outputs=[output])
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, epochs=5, validation_split=0.2, verbose=1, batch_size=100)

# -----
# GRU Bidirectional 

from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import GRU

embed_size=10
rnn_cell_size = 128

sequence_input = Input(shape=(n_steps_in, ))
embedded_sequences = Embedding(k_features, embed_size, input_length=n_steps_in)(sequence_input)
gru_1 = Bidirectional(GRU(rnn_cell_size, return_sequences = True))(embedded_sequences)
outputs, forward_h, backward_h = Bidirectional(GRU(rnn_cell_size, return_sequences=True, return_state=True))(gru_1)
# concatenate the hidden states from each RNN
state_h = Concatenate()([forward_h, backward_h])
context_vector, attention_weights = BahdanauAttention(10)(state_h, outputs)
repeat = RepeatVector(n_steps_out)(context_vector)
gru_2 = Bidirectional(GRU(rnn_cell_size, return_sequences = True))(repeat)
output = TimeDistributed(Dense(k_features, activation='softmax')) (gru_2)

model = keras.models.Model(inputs=[sequence_input], outputs=[output])
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, y, epochs=5, validation_split=0.2, verbose=1, batch_size=100)