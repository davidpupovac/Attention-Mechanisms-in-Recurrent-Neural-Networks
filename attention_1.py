# Python version: 3.7.7
# Tensorflow-gpu version: 1.14.0
# Keras version: 2.2.4-tf


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
"""

                                Attention - teacher forcing 
                   
                         Explicitly defined attention mechanisms
                                
"""
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------


import tensorflow as tf
import numpy as np
import unicodedata
import re

# ----
# If you run the following code and get error 
# UnknownError: Fail to find the dnn implementation. [Op:CudnnRNN]
# first run the following:
    
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

# ----
# Define text processing functions 

# normalizing strings, filtering unwanted tokens, adding space before punctuation
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def normalize_string(s):
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s

# -----------------------------------------------------------------------------
# data

raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('Your idea is not entirely crazy.', "Votre idée n'est pas complètement folle."),
    ("A man's worth lies in what he is.", "La valeur d'un homme réside dans ce qu'il est."),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ("All three of you need to do that.", "Vous avez besoin de faire cela, tous les trois."),
    ("Are you giving me another chance?", "Me donnez-vous une autre chance ?"),
    ("Both Tom and Mary work as models.", "Tom et Mary travaillent tous les deux comme mannequins."),
    ("Can I have a few minutes, please?", "Puis-je avoir quelques minutes, je vous prie ?"),
    ("Could you close the door, please?", "Pourriez-vous fermer la porte, s'il vous plaît ?"),
    ("Did you plant pumpkins this year?", "Cette année, avez-vous planté des citrouilles ?"),
    ("Do you ever study in the library?", "Est-ce que vous étudiez à la bibliothèque des fois ?"),
    ("Don't be deceived by appearances.", "Ne vous laissez pas abuser par les apparences."),
    ("Excuse me. Can you speak English?", "Je vous prie de m'excuser ! Savez-vous parler anglais ?"),
    ("Few people know the true meaning.", "Peu de gens savent ce que cela veut réellement dire."),
    ("Germany produced many scientists.", "L'Allemagne a produit beaucoup de scientifiques."),
    ("Guess whose birthday it is today.", "Devine de qui c'est l'anniversaire, aujourd'hui !"),
    ("He acted like he owned the place.", "Il s'est comporté comme s'il possédait l'endroit."),
    ("Honesty will pay in the long run.", "L'honnêteté paye à la longue."),
    ("How do we know this isn't a trap?", "Comment savez-vous qu'il ne s'agit pas d'un piège ?"),
    ("I can't believe you're giving up.", "Je n'arrive pas à croire que vous abandonniez."),
)

# ----
# Clean up the raw data

# Split the data into two separate lists, each containing its own sentences.
raw_data_en, raw_data_fr = list(zip(*raw_data))
raw_data_en, raw_data_fr = list(raw_data_en), list(raw_data_fr)

# Then  apply the functions above and add two special tokens: <start> and <end>:
raw_data_en = ['<start> ' +normalize_string(data) + '<end>' for data in raw_data_en]
raw_data_fr = ['<start> ' + normalize_string(data) + '<end>' for data in raw_data_fr]
 
# ----
# English data preprocessing

# Tokenize the data
inp_lang = tf.keras.preprocessing.text.Tokenizer(filters='')
inp_lang.fit_on_texts(raw_data_en)
 
# Converte raw English sentences to integer sequences:
input_tensor = inp_lang.texts_to_sequences(raw_data_en)
# pad zeros so that all sequences have the same length
input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, padding='post')

# ----
# French  data preprocessing

targ_lang = tf.keras.preprocessing.text.Tokenizer(filters='')

targ_lang.fit_on_texts(raw_data_fr)

target_tensor = targ_lang.texts_to_sequences(raw_data_fr) # decoder data in
target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, padding='post')

# ------------------------------------------------------------------------------
# model basics

EMBEDDING_SIZE = 32
LSTM_SIZE = 64
BATCH_SIZE = 5
BUFFER_SIZE = len(input_tensor)
EPOCHS = 250
steps_per_epoch = len(input_tensor)//BATCH_SIZE

vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

# ----
# create an instance of tf.data.Dataset:
    
dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE) # one data less here
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

# ------------------------------------------------------------------------------
# Encoder

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size, batch_sz): 
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(
            lstm_size, return_sequences=True, return_state=True)

    def call(self, sequence, hidden):  
        embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=hidden) 

        return output, state_h 

    def initialize_hidden_state(self): # create class methods - for inital hidden state (zero matrix) -
        return (tf.zeros([self.batch_sz, self.lstm_size]), 
                tf.zeros([self.batch_sz, self.lstm_size]))

# ----
encoder = Encoder(vocab_inp_size, EMBEDDING_SIZE, LSTM_SIZE, BATCH_SIZE) # define output of encoder

# sample input
example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape

sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, LSTM units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, LSTM units) {}'.format(sample_hidden.shape))

# ------------------------------------------------------------------------------
# Define Bahdanau attention

class BahdanauAttention(tf.keras.layers.Layer):

  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, dec_hidden, enc_output): 
      
    # encoder hiden states or VALUES = enc_output 
    # decoder's s_t or QUERY = dec_hidden
        
    # Add additional dimension inserted at index axis:
    # we are doing this to broadcast addition along the time axis to calculate the score
    # hidden hidden state shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    dec_hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)
    
    # score shape == (batch_size, max_length, 1)
    score = self.V(tf.nn.tanh(
        self.W1(dec_hidden_with_time_axis) + self.W2(enc_output)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * enc_output 
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

# ----
# check 

attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, lstm_size) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

# ------------------------------------------------------------------------------
# Decoder

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(
            lstm_size, return_sequences=True, return_state=True)
        
        
        self.fc = tf.keras.layers.Dense(vocab_size) # add dense layee
        
        self.attention = BahdanauAttention(self.lstm_size)
    
    def call(self, sequence, hidden, enc_outputs):
        # enc_outputs shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_outputs)

        # sequence shape after passing through embedding == (batch_size, 1, embedding_dim)
        embed = self.embedding(sequence)

        # sequence shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        embed = tf.concat([tf.expand_dims(context_vector, 1), embed], axis=-1)
        # concat(embedding output, context vector)
        
        # passing the concatenated vector to the GRU
        # output, state = self.gru(x)
        output, state_h, state_c = self.lstm(embed)
    
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        sequence = self.fc(output)

        return sequence, state_h, attention_weights

# ----    
decoder = Decoder(vocab_tar_size, EMBEDDING_SIZE, LSTM_SIZE, BATCH_SIZE) # define output of decoder

# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, lstm_size) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, lstm_size) {}'.format(sample_hidden.shape))    
    
# ------------------------------------------------------------------------------
# Define loss function 

# Since we padded zeros into the sequences, do not take zeros into account when computing the loss:
def loss_function(targets, logits):
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(targets, 0)) # masking
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss 

# ----
# Define optimizer
optimizer = tf.keras.optimizers.Adam()

# ----
@tf.function # not sure what this is...


def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))
  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

# -----
# Train

import time

for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss

    if batch % 100 == 0:
      print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))

  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

# ------------------------------------------------------------------------------
# Evaluate, plotting the attention weights, translate

# The evaluate function is similar to the training loop, except we don't use teacher forcing here. 
# The input to the decoder at each time step is its previous predictions along with the
# hidden state and the encoder output. Stop predicting when the model predicts the end token.
# And store the attention weights for every time step.

max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

def preprocess_sentence(w):
  w = unicode_to_ascii(w.lower().strip())

  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
  w = w.strip()
  # w = '<start> ' + w + ' <end>'
  return w

# -----
# Method for inference purpose:

# Basically a forward pass, but instead of target sequences, we will feed in the <start> token.
# Every next time step will take the output of the last time step as input until we hit the <end>
# token or the output sequence has exceed a specific length:

def evaluate(sentence):
  attention_plot = np.zeros((max_length_targ, max_length_inp))

  sentence = preprocess_sentence(sentence)

  inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  # hidden = [tf.zeros((1, LSTM_SIZE))]
  hidden =[tf.zeros((1, LSTM_SIZE)), # specific for LSTM!!!
           tf.zeros((1, LSTM_SIZE))]
  
  enc_out, enc_hidden = encoder(inputs, hidden) 

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input, 
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += targ_lang.index_word[predicted_id] + ' '

    if targ_lang.index_word[predicted_id] == '<end>':
        return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot

# -----
# function for plotting the attention weights

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_attention(attention, sentence, predicted_sentence):
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()
  
# -----
def translate(sentence):
  result, sentence, attention_plot = evaluate(sentence)

  print('Input: %s' % (sentence))
  print('Predicted translation: {}'.format(result))

  attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
  plot_attention(attention_plot, sentence.split(' '), result.split(' '))
  
# -----
preprocess_sentence(u'Honesty will pay in the long run.')
translate(u'Honesty will pay in the long run.')


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
#                         Luong Attention
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# data

raw_data = (
    ('What a ridiculous concept!', 'Quel concept ridicule !'),
    ('Your idea is not entirely crazy.', "Votre idée n'est pas complètement folle."),
    ("A man's worth lies in what he is.", "La valeur d'un homme réside dans ce qu'il est."),
    ('What he did is very wrong.', "Ce qu'il a fait est très mal."),
    ("All three of you need to do that.", "Vous avez besoin de faire cela, tous les trois."),
    ("Are you giving me another chance?", "Me donnez-vous une autre chance ?"),
    ("Both Tom and Mary work as models.", "Tom et Mary travaillent tous les deux comme mannequins."),
    ("Can I have a few minutes, please?", "Puis-je avoir quelques minutes, je vous prie ?"),
    ("Could you close the door, please?", "Pourriez-vous fermer la porte, s'il vous plaît ?"),
    ("Did you plant pumpkins this year?", "Cette année, avez-vous planté des citrouilles ?"),
    ("Do you ever study in the library?", "Est-ce que vous étudiez à la bibliothèque des fois ?"),
    ("Don't be deceived by appearances.", "Ne vous laissez pas abuser par les apparences."),
    ("Excuse me. Can you speak English?", "Je vous prie de m'excuser ! Savez-vous parler anglais ?"),
    ("Few people know the true meaning.", "Peu de gens savent ce que cela veut réellement dire."),
    ("Germany produced many scientists.", "L'Allemagne a produit beaucoup de scientifiques."),
    ("Guess whose birthday it is today.", "Devine de qui c'est l'anniversaire, aujourd'hui !"),
    ("He acted like he owned the place.", "Il s'est comporté comme s'il possédait l'endroit."),
    ("Honesty will pay in the long run.", "L'honnêteté paye à la longue."),
    ("How do we know this isn't a trap?", "Comment savez-vous qu'il ne s'agit pas d'un piège ?"),
    ("I can't believe you're giving up.", "Je n'arrive pas à croire que vous abandonniez."),
)

# ----
# Clean up the raw data

# Split the data into two separate lists, each containing its own sentences.
raw_data_en, raw_data_fr = list(zip(*raw_data))
raw_data_en, raw_data_fr = list(raw_data_en), list(raw_data_fr)

# Then  apply the functions above and add two special tokens: <start> and <end>:
raw_data_en = [normalize_string(data) for data in raw_data_en]
raw_data_fr_in = ['<start> ' + normalize_string(data) for data in raw_data_fr]
raw_data_fr_out = [normalize_string(data) + ' <end>' for data in raw_data_fr]    


# ----
# English data preprocessing

# Tokenize the data, i.e. convert the raw strings into integer sequences
en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

# and create vocabulary
en_tokenizer.fit_on_texts(raw_data_en)
 
# The tokenizer created its own vocabulary as well as conversion dictionaries. 
print(en_tokenizer.word_index)

# Converte draw English sentences to integer sequences:
data_en = en_tokenizer.texts_to_sequences(raw_data_en)

# pad zeros so that all sequences have the same length
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en, padding='post')

# ----
# French  data preprocessing

fr_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

fr_tokenizer.fit_on_texts(raw_data_fr_in)
fr_tokenizer.fit_on_texts(raw_data_fr_out)

data_fr_in = fr_tokenizer.texts_to_sequences(raw_data_fr_in) # decoder data in
data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in, padding='post')

data_fr_out = fr_tokenizer.texts_to_sequences(raw_data_fr_out) # decoder data out
data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out, padding='post')

# ----
# create an instance of tf.data.Dataset:
dataset = tf.data.Dataset.from_tensor_slices((data_en, data_fr_in, data_fr_out))
dataset = dataset.shuffle(20).batch(5)

# ------------------------------------------------------------------------------
# model basics

NUM_EPOCHS = 250
BATCH_SIZE = 5
EMBEDDING_SIZE = 32
LSTM_SIZE = 64

en_vocab_size = len(en_tokenizer.word_index) + 1
fr_vocab_size = len(fr_tokenizer.word_index) + 1

# ------------------------------------------------------------------------------
# Encoder

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, lstm_size): 
        super(Encoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(
            lstm_size, return_sequences=True, return_state=True)

    def call(self, sequence, states):  
        embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=states)

        return output, state_h, state_c

    def init_states(self, batch_size): 
        return (tf.zeros([batch_size, self.lstm_size]),
                tf.zeros([batch_size, self.lstm_size]))

encoder = Encoder(en_vocab_size, EMBEDDING_SIZE, LSTM_SIZE) 

# ------------------------------------------------------------------------------
# Luong Attention


class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size):
        super(LuongAttention, self).__init__()
        self.wa = tf.keras.layers.Dense(rnn_size)
        
    def call(self, decoder_output, encoder_output):
        
        # Dot score: h_t (dot) Wa (dot) h_s
        # decoder_output shape: (batch_size, 1, rnn_size)
        # encoder_output shape: (batch_size, max_len, rnn_size)
        # score will have shape: (batch_size, 1, max_len)
        
        score = tf.matmul(decoder_output, self.wa(encoder_output), transpose_b=True) 
        
        alignment = tf.nn.softmax(score, axis=2)
        
        # context vector = weighted average of the encoder’s output: 
        # (i.e. dot product of the alignment vector and the encoder’s output)
        context = tf.matmul(alignment, encoder_output)

        return context, alignment

# ------------------------------------------------------------------------------
# Decoder

# we need to create an attention object when creating the decoder:
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, rnn_size):
        super(Decoder, self).__init__()
        
        # Create a LuongAttention object
        self.attention = LuongAttention(rnn_size)

        self.rnn_size = rnn_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(
            rnn_size, return_sequences=True, return_state=True)
        
        self.wc = tf.keras.layers.Dense(rnn_size, activation='tanh')
        self.ws = tf.keras.layers.Dense(vocab_size)
        
    def call(self, sequence, state, encoder_output):

        embed = self.embedding(sequence)
        
        # Therefore, the lstm_out has shape (batch_size, 1, rnn_size)
        lstm_out, state_h, state_c = self.lstm(embed, initial_state=state)
            
        # Use self.attention to compute the context and alignment vectors
        # context vector's shape: (batch_size, 1, rnn_size)
        # alignment vector's shape: (batch_size, 1, source_length)
        context, alignment = self.attention(lstm_out, encoder_output)
     
        lstm_out = tf.concat([tf.squeeze(context, 1), tf.squeeze(lstm_out, 1)], 1)
        
        # lstm_out now has shape (batch_size, rnn_size)
        lstm_out = self.wc(lstm_out)
        
        # Finally, it is converted back to vocabulary space: (batch_size, vocab_size)
        logits = self.ws(lstm_out)

        return logits, state_h, state_c, alignment

decoder = Decoder(fr_vocab_size, EMBEDDING_SIZE, LSTM_SIZE) # define output of decoder

# ------------------------------------------------------------------------------
# Train

def train_step(source_seq, target_seq_in, target_seq_out, en_initial_states):
    loss = 0
    with tf.GradientTape() as tape:
        en_outputs = encoder(source_seq, en_initial_states)
        en_states = en_outputs[1:]
        de_state_h, de_state_c = en_states
        
        # We need to create a loop to iterate through the target sequences
        for i in range(target_seq_out.shape[1]):
            # Input to the decoder must have shape of (batch_size, length)
            # so we need to expand one dimension
            decoder_in = tf.expand_dims(target_seq_in[:, i], 1)
            logit, de_state_h, de_state_c, _ = decoder(
                decoder_in, (de_state_h, de_state_c), en_outputs[0])
            
            # The loss is now accumulated through the whole batch
            loss += loss_func(target_seq_out[:, i], logit)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss / target_seq_out.shape[1]

# ----
# Do the same to the predict function.
    
def predict(test_source_text=None):
    if test_source_text is None:
        test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]
    print(test_source_text)
    test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
    print(test_source_seq)

    en_initial_states = encoder.init_states(1)
    en_outputs = encoder(tf.constant(test_source_seq), en_initial_states)

    de_input = tf.constant([[fr_tokenizer.word_index['<start>']]])
    de_state_h, de_state_c = en_outputs[1:]
    out_words = []
    alignments = []

    while True:
        de_output, de_state_h, de_state_c, alignment = decoder(
            de_input, (de_state_h, de_state_c), en_outputs[0])
        de_input = tf.expand_dims(tf.argmax(de_output, -1), 0)
        out_words.append(fr_tokenizer.index_word[de_input.numpy()[0][0]])
        
        alignments.append(alignment.numpy())

        if out_words[-1] == '<end>' or len(out_words) >= 20:
            break

    print(' '.join(out_words))
    return np.array(alignments), test_source_text.split(' '), out_words

# -----------------------------------------------------------------------------
# Define a loss function 

# Since we padded zeros into the sequences, do not take zeros into account when computing the loss:
def loss_func(targets, logits):
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss 

# Define optimizer
optimizer = tf.keras.optimizers.Adam()

# ----
# Training loop

for e in range(NUM_EPOCHS):
    en_initial_states = encoder.init_states(BATCH_SIZE)

    for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
        loss = train_step(source_seq, target_seq_in,
                          target_seq_out, en_initial_states)

    print('Epoch {} Loss {:.4f}'.format(e + 1, loss.numpy()))
    
    try: # if you want some visualistions run this part of code as well
        predict()
    except Exception:
      continue