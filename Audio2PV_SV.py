# Auto-Encoder
import numpy
import tensorflow as tf
from tensorflow.contrib import rnn

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
shuffle_buffer_size = 20000
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
speaker_vec_len = 32 # hidden layer num of features
phonetic_vec_len = 320
D_num_hidden = 128

##############################################################
#                        Model                               #
##############################################################
Ep_Input= tf.placeholder("float", [None, timesteps, num_input])
Eps_Input= tf.placeholder("float", [None, timesteps, num_input])
Es_Input= tf.placeholder("float", [None, timesteps, num_input])


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
        output = states.h
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
        output = states.h
        return tf.matmul(output, weights['out']) + biases['out']

def RNN_decoder(x, weights, biases):
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) as de:
        # Define a lstm cell with tensorflow
        lstm_cell_decoder = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        # Get lstm cell output
        outputs, states = tf.nn.dynamic_rnn(lstm_cell_decoder, x, dtype=tf.float32)
        train_decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell,initial_state=decoder_intial_state)
        tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder)
        # Linear activation, using rnn inner loop last output
        output = states.h
        return tf.matmul(output, weights['out']) + biases['out']

tf.Graph()

p1_vec = RNN_phone(Ep_Input, weights_Ep, biases_Ep)
p2_vec = RNN_phone(Eps_Input, weights_Ep, biases_Ep)
p3_vec = RNN_phone(Es_Input, weights_Ep, biases_Ep)
s1_vec = RNN_speaker(Eps_Input, weights_Es, biases_Es)
s2_vec = RNN_speaker(Es_Input, weights_Es, biases_Es)
ps_vec = tf.concat([p2_vec, s1_vec], axis=1)
output = RNN_decoder(ps_vec, weights_De, biases_De)

AE_loss = tf.reduce_mean(tf.square(output - Ep_Input))

# Speaker Discriminator
D_W1 = tf.Variable(xavier_init([phonetic_vec_len, D_num_hidden ]))
D_b1 = tf.Variable(tf.zeros(shape=[D_num_hidden ]))

D_W2 = tf.Variable(xavier_init([D_num_hidden , 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]
theta_G = [weights_Ep, biases_Ep]

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


# first, concat 2 phonetic vector feed in discriminator
Same_speaker = tf.concat(p1_vec, s2_vec)
Diff_speaker = tf.concat(p2_vec, p3_vec) ##changing the order make any sense??

D_same, D_logit_same = discriminator(Same_speaker)
D_diff, D_logit_diff = discriminator(Diff_speaker)

# then, train the speaker discriminator

D_loss = -tf.reduce_mean(tf.log(D_same) + tf.log(1. - D_diff))
G_loss = -tf.reduce_mean(tf.log(D_diff))

# Make Speaker vector as close as possible
L2_loss = tf.reduce_mean(tf.square(s1_vec - s2_vec))

loss_1 = AE_loss + L2_loss + D_loss
loss_2 = G_loss

loss1_solver = tf.train.AdamOptimizer().minimize(loss_1, var_list = theta_D)
loss2_solver = tf.train.AdamOptimizer().minimize(loss_2, var_list = theta_G)
AE_solver = tf.train.AdamOptimizer().minimize(AE_loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

#############################################################
#                    Training Process                       #
#############################################################
signal =sess.run(next_element)['matrix']
print("HERE")
for it in range(10000):
    prev_element = signal
    signal = sess.run(next_element)['matrix']
    rand_spk = sess.run(next_spk)['matrix']

    _, loss1_curr = sess.run([loss1_solver, loss_1], feed_dict={Ep_Input:rand_spk,Eps_Input:signal,Es_Input:prev_element})
    _, loss2_curr = sess.run([loss2_solver, loss_2], feed_dict={Ep_Input:rand_spk,Eps_Input:signal,Es_Input:prev_element})

    if it % 2 == 0:
        _, AE_loss_curr = sess.run([AE_solver, AE_loss], feed_dict={Ep_Input:rand_spk,Eps_Input:signal,Es_Input:prev_element})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('loss1: {:.4}'.format(loss1_curr))
        print('loss2: {:.4}'.format(loss2_curr))
        print('AE_loss: {:.4}'.format(sess.run(E_loss)))
        print('L2_loss: {:.4}'.format(sess.run(L2_loss)))
        print('D_loss: {:.4}'.format(sess.run(D_loss)))
        print('G_loss: {:.4}'.format(G_loss))
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)

