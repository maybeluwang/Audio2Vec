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

# Load dataset
filename = 'test.tfrecord'
dataset = tf.data.TFRecordDataset(filename)

# Parameters for dataset
shuffle_buffer_size = 1000
batch_size = 32
epoch = 10

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
dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size).repeat(epoch)

# Create an iterator that can easily train
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
sess = tf.InteractiveSession()

###############################################################
#                     Training parameters                     #
###############################################################
# Autoencoder training parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 512
speaker_vec_len = 32 # hidden layer num of features
phonetic_vec_len = 320
num_classes = 10 # MNIST total classes (0-9 digits)

##############################################################
#                        Model                               #
##############################################################
Ep_Input= tf.placeholder("float", [None, timesteps, num_input])
Es_Input= tf.placeholder("float", [None, timesteps, num_input])


weights_Ep = {'out': tf.Variable(tf.random_normal([num_hidden, phonetic_vec_len]))}
biases_Ep = {'out': tf.Variable(tf.random_normal([phonetic_vec_len]))}
weights_Es = {'out': tf.Variable(tf.random_normal([num_hidden, speaker_vec_len]))}
biases_Es = {'out': tf.Variable(tf.random_normal([speaker_vec_len]))}
weights_De = {'out': tf.Variable(tf.random_normal([num_hidden, 39]))}
biases_De = {'out': tf.Variable(tf.random_normal([39]))}



def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

p1_vec = RNN(Ep_Input1, weights_Ep, biases_Ep)
s1_vec = RNN(Es_Input1, weights_Es, biases_Ep)
s2_vec = RNN(Es_Input2, weights_Es, biases_Ep)
ps_vec = tf.concat(p_vec1, s_vec1)
ps_vec_seq = tf.tile(ps_vec, multiples)
output = RNN(ps_vec_seq, weights_De, biases_De)

AE_loss = tf.reduce_mean(tf.square(output - Ep_Input))

# Speaker Discriminator
D_W1 = tf.Variable(xavier_init([784, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


# first, concat 2 phonetic vector feed in discriminator
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

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#############################################################
#                    Training Process                       #
#############################################################
for it in range(1000):
    word = sess.run(next_element)
    _, loss1_curr = sess.run([loss1_solver, loss_1], feed_dict{X: X_data, Z: sample_Z(mb_size, Z_dim)})
    _, loss2_curr = sess.run([loss2_solver, loss_2], feed_dict{})

    if it % 2 == 0:
        _, AE_loss_curr = sess.run([AE_solver, AE_loss], feed_dict{})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('loss1: {:.4}'.format(loss1_curr))
        print('loss2: {:.4}'.format(loss2_curr))
        print('AE_loss: {:.4}'.format(AE_loss))
        print('L2_loss: {:.4}'.format(L2_loss))
        print('D_loss: {:.4}'.format(D_loss))
        print('G_loss: {:.4}'.format(G_loss))

