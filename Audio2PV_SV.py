# Auto-Encoder
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.layers.core import Dense

###############################################################
#                      Preprocessing                          #
###############################################################
# Data Preprocessing is done by DataReader                    #
# We just simmply load tensorflow API dataset                 #
# By tensorflow API, read data pipeline, fast, and easily     #
# Only run in python >= 3.4 and tensorflow >= 1.3             #
###############################################################
# Load tensorflow API dataset
filename = 'test.tfrecord'

dataset = tf.data.TFRecordDataset(filename)
dataset_diffspk = tf.data.TFRecordDataset(filename)


# Parameters
shuffle_buffer_size = 50000
batch_size = 16
epoch = 1

# by tensorflow API, read data parallel, fast, and easily
# only run in python >= 3.4 and tensorflow >= 1.3
def parse_function(serialized_example):
    features = tf.parse_single_example(serialized_example,
        features={
            'speaker': tf.FixedLenFeature( shape = (),dtype = tf.string),
            'word': tf.FixedLenFeature(shape=(1,), dtype = tf.int64),
            'phoneList': tf.VarLenFeature( tf.int64),
            'matrix': tf.VarLenFeature( tf.float32),
            'matrix_shape':tf.FixedLenFeature(shape=(2,),dtype = tf.int64),
        })
    matrix = tf.sparse_tensor_to_dense(features['matrix'])
    features['matrix_shape'] = tf.cast(features['matrix_shape'], tf.int32)
    features['matrix'] = tf.reshape(matrix,features['matrix_shape'])
    return features


dataset = dataset.map(parse_function)
dataset_diffspk = dataset_diffspk.map(parse_function)

dataset = dataset.batch(batch_size).repeat(epoch)
dataset_diffspk = dataset_diffspk.shuffle(shuffle_buffer_size).batch(batch_size).repeat(epoch)

# Create an iterator that can easily train
iterator = dataset.make_one_shot_iterator()
iterator_diffspk = dataset_diffspk.make_one_shot_iterator()
next_spk = iterator_diffspk.get_next()
next_element = iterator.get_next()

###############################################################
#                     Training parameters                     #
###############################################################
# Autoencoder training parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 39 # MFCC inputs (dim = 39)
timesteps = 400 # timesteps
num_hidden = 512
speaker_vec_len = 12 # hidden layer num of features
phonetic_vec_len = 320
D_num_hidden = 128

##############################################################
#                        Model                               #
##############################################################
Ep_Input= tf.placeholder("float", [None, timesteps, num_input])
Eps_Input= tf.placeholder("float", [None, timesteps, num_input])
Es_Input= tf.placeholder("float", [None, timesteps, num_input])
Sequence_Length = tf.placeholder("int32", [None])

weights_Ep = {'out': tf.Variable(tf.random_normal([num_hidden, phonetic_vec_len]))}
biases_Ep = {'out': tf.Variable(tf.random_normal([phonetic_vec_len]))}
weights_Es = {'out': tf.Variable(tf.random_normal([num_hidden, speaker_vec_len]))}
biases_Es = {'out': tf.Variable(tf.random_normal([speaker_vec_len]))}
weights_De = {'out': tf.Variable(tf.random_normal([num_hidden, 39]))}
biases_De = {'out': tf.Variable(tf.random_normal([39]))}

def RNN_phone(x, weights, biases):
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
    with tf.variable_scope('phone',reuse=tf.AUTO_REUSE) as ph:
        # Define a lstm cell with tensorflow
        lstm_cell_phone = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        # Get lstm cell output
        outputs, states = tf.nn.dynamic_rnn(lstm_cell_phone, x, dtype=tf.float32)#, time_major=True)
        # Linear activation, using rnn inner loop last output
        output = outputs[:,-1,:]
        return tf.matmul(output, weights['out']) + biases['out']

def RNN_speaker(x, weights, biases):
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
    with tf.variable_scope('speaker', reuse=tf.AUTO_REUSE) as sp:
        # Define a lstm cell with tensorflow
        lstm_cell_speaker = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        # Get lstm cell output
        outputs, states = tf.nn.dynamic_rnn(lstm_cell_speaker, x, dtype=tf.float32)
        # Linear activation, using rnn inner loop last output
        output = outputs[:,-1,:]
        return tf.matmul(output, weights['out']) + biases['out']

def RNN_decoder(x, weights, biases):
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) as de:
        # Define a lstm cell with tensorflow
        lstm_cell_decoder = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        output_layer = Dense(units=39)
        # Get lstm cell output
        #outputs, states = tf.nn.dynamic_rnn(lstm_cell_decoder, x, dtype=tf.float32)
        decoder_initial_state = lstm_cell_decoder.zero_state(tf.shape(x)[0], dtype=tf.float32)

        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=x,sequence_length = Sequence_Length,time_major=False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell = lstm_cell_decoder,initial_state=decoder_initial_state,helper=training_helper,output_layer= output_layer)
        decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,maximum_iterations=timesteps)
        # Linear activation, using rnn inner loop last output
        #output = outputs[:,-1,:]
        return decoder_outputs.rnn_output#tf.matmul(tf.transpose(decoder_outputs[0],perm = [1,0,2]), weights['out']) + biases['out']

tf.Graph()

p1_vec = RNN_phone(Ep_Input, weights_Ep, biases_Ep)
p2_vec = RNN_phone(Eps_Input, weights_Ep, biases_Ep)
p3_vec = RNN_phone(Es_Input, weights_Ep, biases_Ep)
s1_vec = RNN_speaker(Eps_Input, weights_Es, biases_Es)
s2_vec = RNN_speaker(Es_Input, weights_Es, biases_Es)
ps_vec = tf.concat([p2_vec, s1_vec], axis=1)
ps_vec_seq = tf.tile([ps_vec], [timesteps,1,1])
ps_vec_seq = tf.transpose(ps_vec_seq, perm=[1,0,2])
output = RNN_decoder(ps_vec_seq, weights_De, biases_De)

AE_loss = tf.reduce_mean(tf.square(output - Eps_Input))
# Speaker Discriminator
D_W1 = {'dis': tf.Variable(tf.random_normal([ phonetic_vec_len*2,39]))}
D_b1 = tf.Variable(tf.zeros(shape=[39 ]))
D_W2 = {'dis': tf.Variable(tf.random_normal([ 39,1]))}


D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]
theta_G = [weights_Ep, biases_Ep]

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1['dis']) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2['dis']) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


# first, concat 2 phonetic vector feed in discriminator
Same_speaker = tf.concat([p1_vec, p2_vec],axis=1)
Diff_speaker = tf.concat([p2_vec, p3_vec],axis=1) ##changing the order make any sense??

D_same, D_logit_same = discriminator(Same_speaker)
D_diff, D_logit_diff = discriminator(Diff_speaker)

# then, train the speaker discriminator

D_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(D_same, 1e-10, 1.0)) + tf.log(tf.clip_by_value(1. - D_diff, 1e-10, 1.0)))
G_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(D_diff, 1e-10, 1.0)))

# Make Speaker vector as close as possible
L2_loss = tf.reduce_mean(tf.square(s1_vec - s2_vec))

loss_1 = AE_loss + L2_loss + 10*D_loss
loss_2 = G_loss

loss1_solver_D = tf.train.AdamOptimizer().minimize(loss_1, var_list = theta_D)
loss1_solver = tf.train.AdamOptimizer().minimize(loss_1)
loss2_solver = tf.train.AdamOptimizer().minimize(loss_2, var_list = theta_G)

AE_solver = tf.train.AdamOptimizer().minimize(AE_loss)
D_solver = tf.train.AdamOptimizer().minimize(D_loss)
L2_solver = tf.train.AdamOptimizer().minimize(L2_loss)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

#############################################################
#                    Training Process                       #
#############################################################
signal =sess.run(next_element)['matrix']
sequence_length = np.ones((batch_size), dtype=int) * timesteps

for it in range(10000):
    prev_element = signal
    signal = sess.run(next_element)['matrix']
    rand_spk = sess.run(next_spk)['matrix']
    _, AE_loss_curr = sess.run([AE_solver, AE_loss], feed_dict={Ep_Input:rand_spk, Eps_Input:signal, Es_Input:prev_element, Sequence_Length:sequence_length})

    
    prev_element = signal
    signal = sess.run(next_element)['matrix']
    rand_spk = sess.run(next_spk)['matrix']
 
    _, loss1_curr, L2_loss_curr, D_loss_curr, AE_loss_curr = sess.run([loss1_solver, loss_1, L2_loss, D_loss, AE_loss], feed_dict={Ep_Input:rand_spk, Eps_Input:signal, Es_Input:prev_element, Sequence_Length:sequence_length})
    _, loss2_curr = sess.run([loss2_solver, loss_2], feed_dict={Ep_Input:rand_spk, Eps_Input:signal, Es_Input:prev_element, Sequence_Length:sequence_length})
    
    if it % 50 == 0:
        print('Iter: {}'.format(it))
        print('loss1: {:.4}'.format(loss1_curr))
        print('loss2: {:.4}'.format(loss2_curr))
        print('AE_loss: {:.4}'.format(AE_loss_curr))
        print('L2_loss: {:.4}'.format(L2_loss_curr))
        print('D_loss: {:.4}'.format(D_loss_curr))
        save_path = saver.save(sess, "./tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)

