# Python version: 3.7.7
# Tensorflow-gpu version: 1.14.0
# Keras version: 2.2.4-tf


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
"""

                            Attention using tfa.seq2seq addon  
                            
"""
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------


# If you run the following code and get error 
# UnknownError: Fail to find the dnn implementation. [Op:CudnnRNN]
# first run the following:
    
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

# -----------------------------------------------------------------------------

import numpy as np
from random import randint
from numpy import array

# Data description: 
# Input is a sequence of n_in numbers. Target is first n_out elements
# of the input sequence in the reversed order 

# generate a sequence of random integers
def gen_sequence(length, n_unique): 
	return [randint(1, n_unique) for _ in range(length)]


def gen_dataset(n_in, n_out, cardinality, n_samples):
	X, y = list(), list()
	for _ in range(n_samples):
		# generate source sequence
		source = gen_sequence(n_in, cardinality)
        
		# define target sequence:
		# take first n elements of the source sequence as the target sequence and reverse them
		target = source[:n_out] # these values will be passed to encoder inputs 
		target.reverse() # the values are targets
        
		target = [cardinality + 1] + target + [cardinality + 2] # add <start> and <end> tokens
		source = [cardinality + 1] + source + [cardinality + 2]
		
        # store (create inputs)
		X.append(source)
		y.append(target)
	return array(X), array(y)

k_features = 50 # length of actual vocabulary
n_steps_in = 10 # time steps in 
n_steps_out = 6 # time steps out

X, y = gen_dataset(n_steps_in, n_steps_out, k_features, 10000)
print(X.shape, y.shape)

# -----------------------------------------------------------------------------
# Model Parameters

import tensorflow as tf
import tensorflow_addons as tfa # requires TensorFlow version >= 2.1.0
from sklearn.model_selection import train_test_split

X_train,  X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2)

BATCH_SIZE = 64
BUFFER_SIZE = len(X_train)
steps_per_epoch = BUFFER_SIZE//BATCH_SIZE
embedding_dims = 256
rnn_units = 1024
dense_units = 1024 # number of units in attention 
Dtype = tf.float32  # used to initialize DecoderCell Zero state

# -----------------------------------------------------------------------------

def max_len(tensor):
    #print( np.argmax([len(t) for t in tensor]))
    return max( len(t) for t in tensor)

Tx = max_len(X) # maximal length of sequence; here n_steps_in
Ty = max_len(y) # maximal length of sequence; here n_steps_out


input_vocab_size = k_features+2+1 # size of vocabulary, add 2 for <start> and <end> tokens
output_vocab_size = k_features+2+1
dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
example_X, example_Y = next(iter(dataset)) # batch example 

print(example_X.shape) 
print(example_Y.shape)

# -----------------------------------------------------------------------------
# Define encoder and decoder 

# Encoder
class EncoderNetwork(tf.keras.Model):
    def __init__(self,input_vocab_size,embedding_dims, rnn_units ):
        super().__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size,
                                                           output_dim=embedding_dims)
        self.encoder_rnnlayer = tf.keras.layers.LSTM(rnn_units,return_sequences=True, 
                                                     return_state=True )
    
# Decoder
class DecoderNetwork(tf.keras.Model):
    def __init__(self,output_vocab_size, embedding_dims, rnn_units):
        super().__init__()
        self.decoder_embedding = tf.keras.layers.Embedding(input_dim=output_vocab_size,
                                                           output_dim=embedding_dims) 
        self.dense_layer = tf.keras.layers.Dense(output_vocab_size)
        self.decoder_rnncell = tf.keras.layers.LSTMCell(rnn_units) # define decoder_rnncell
        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(dense_units,None,BATCH_SIZE*[Tx])
        self.rnn_cell =  self.build_rnn_cell(BATCH_SIZE)
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler= self.sampler,
                                                output_layer=self.dense_layer)
        
    # define atttention mechanism
    def build_attention_mechanism(self, units,memory, memory_sequence_length):
        return tfa.seq2seq.LuongAttention(units, memory = memory, 
                                          memory_sequence_length=memory_sequence_length)
        # if you want Bahdanau attention:
        # return tfa.seq2seq.BahdanauAttention(units, memory = memory, memory_sequence_length=memory_sequence_length)

    # wrap decodernn cell  
    def build_rnn_cell(self, batch_size ):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnncell, self.attention_mechanism,
                                                attention_layer_size=dense_units)
        return rnn_cell
    
    # define initial state
    def build_decoder_initial_state(self, batch_size, encoder_state,Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size = batch_size, 
                                                                dtype = Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state) 
        return decoder_initial_state

encoderNetwork = EncoderNetwork(input_vocab_size,embedding_dims, rnn_units)
decoderNetwork = DecoderNetwork(output_vocab_size,embedding_dims, rnn_units)
optimizer = tf.keras.optimizers.Adam()

# -----------------------------------------------------------------------------
# Initializing Training functions

def loss_function(y_pred, y):
   
    # shape of y_pred [batch_size, Ty, output_vocab_size] 
    sparsecategoricalcrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                  reduction='none')
    loss = sparsecategoricalcrossentropy(y_true=y, y_pred=y_pred)
    
    mask = tf.logical_not(tf.math.equal(y,0))   #output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask* loss
    
    loss = tf.reduce_mean(loss)
    return loss

decoderNetwork.attention_mechanism.memory_initialized


def train_step(input_batch, output_batch,encoder_initial_cell_state):
    #initialize loss = 0
    loss = 0
    with tf.GradientTape() as tape:
        encoder_emb_inp = encoderNetwork.encoder_embedding(input_batch)
        a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp, 
                                                        initial_state =encoder_initial_cell_state)
        # [last step activations,last memory_state] of encoder passed as input to decoder Network
              
        # Prepare correct Decoder input & output sequence data
        decoder_input = output_batch[:,:-1] # to ignore <end> 
        #compare logits with timestepped +1 version of decoder_input
        decoder_output = output_batch[:,1:] # to ignore <start> 

        # Decoder Embeddings
        decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)

        # Setting up decoder memory from encoder output and Zero State for AttentionWrapperState
        decoderNetwork.attention_mechanism.setup_memory(a)
        decoder_initial_state = decoderNetwork.build_decoder_initial_state(BATCH_SIZE,
                                                                           encoder_state=[a_tx, c_tx],
                                                                           Dtype=tf.float32)
        
        # BasicDecoderOutput        
        outputs, _, _ = decoderNetwork.decoder(decoder_emb_inp,initial_state=decoder_initial_state,
                                               sequence_length=BATCH_SIZE*[Ty-1]) 

        logits = outputs.rnn_output

        # Calculate loss
        loss = loss_function(logits, decoder_output)

    # Returns the list of all layer variables / weights.
    variables = encoderNetwork.trainable_variables + decoderNetwork.trainable_variables  
    # differentiate loss wrt variables
    gradients = tape.gradient(loss, variables)

    # grads_and_vars – List of(gradient, variable) pairs.
    grads_and_vars = zip(gradients,variables)
    optimizer.apply_gradients(grads_and_vars)
    return loss

# -----------------------------------------------------------------------------
# RNN LSTM hidden and memory state initializer

def initialize_initial_state():
        return [tf.zeros((BATCH_SIZE, rnn_units)), tf.zeros((BATCH_SIZE, rnn_units))]

# -----------------------------------------------------------------------------
# Training

epochs = 15
for i in range(1, epochs+1):

    encoder_initial_cell_state = initialize_initial_state()
    total_loss = 0.0

    for ( batch , (input_batch, output_batch)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(input_batch, output_batch, encoder_initial_cell_state)
        total_loss += batch_loss
        if (batch+1)%5 == 0:
            print("total loss: {} epoch {} batch {} ".format(batch_loss.numpy(), i, batch+1))
            
# -----------------------------------------------------------------------------
# Evaluation

# For this we use greedsampler to run through the decoder
# and the final embedding matrix trained on the data is used to generate embeddings

# generate new data

X_test, y_test = gen_dataset(n_steps_in, n_steps_out, k_features, 1)
print(X_test, y_test) 
print(X_test.shape, y_test.shape) 

input_sequences=X_test

inp = tf.convert_to_tensor(input_sequences)

inference_batch_size = input_sequences.shape[0]
encoder_initial_cell_state = [tf.zeros((inference_batch_size, rnn_units)),
                              tf.zeros((inference_batch_size, rnn_units))]

encoder_emb_inp = encoderNetwork.encoder_embedding(inp)
encoder_emb_inp.shape

a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp,
                                                initial_state =encoder_initial_cell_state)

print('a_tx :',a_tx.shape)
print('c_tx :', c_tx.shape)

greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

decoder_instance = tfa.seq2seq.BasicDecoder(cell = decoderNetwork.rnn_cell, sampler = greedy_sampler,
                                            output_layer=decoderNetwork.dense_layer)
decoderNetwork.attention_mechanism.setup_memory(a)

print("decoder_initial_state = [a_tx, c_tx] :",np.array([a_tx, c_tx]).shape)

decoder_initial_state = decoderNetwork.build_decoder_initial_state(inference_batch_size,
                                                                   encoder_state=[a_tx, c_tx],
                                                                   Dtype=tf.float32)

maximum_iterations = n_steps_out

# set emedding
decoder_embedding_matrix = decoderNetwork.decoder_embedding.variables[0] 
print(decoderNetwork.decoder_embedding.variables[0].shape)

start_tokens = tf.fill([1] ,k_features+1) # must be vector
end_token = np.array(k_features+2) # must be scalar

# initialize inference decoder
(first_finished, first_inputs,first_state) = decoder_instance.initialize(decoder_embedding_matrix,
                             start_tokens = start_tokens,
                             end_token=end_token,
                             initial_state = decoder_initial_state)

inputs = first_inputs
state = first_state  

predictions = np.empty((inference_batch_size,0), dtype = np.int32)                                                                             

for j in range(maximum_iterations):
    outputs, next_state, next_inputs, finished = decoder_instance.step(1,inputs,state)
    inputs = next_inputs
    state = next_state
    outputs = np.expand_dims(outputs.sample_id,axis = -1)
    predictions = np.append(predictions, outputs, axis = -1)

# -----------------------------------------------------------------------------
# Final prediction

import itertools

# prediction based on our sentence earlier
print("Input:")
print(input_sequences[:,1:-1])
print("\nOutput:")
for i in range(len(predictions)):
    line = predictions[0,:]
    seq = list(itertools.takewhile( lambda index: index !=2, line))
    print(seq)            
            