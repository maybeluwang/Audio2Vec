import tensorflow as tf

# Load tensorflow API dataset
filename = 'test.tfrecord'

dataset = tf.data.TFRecordDataset(filename)
dataset_diffspk = tf.data.TFRecordDataset(filename)


# Parameters
shuffle_buffer_size = 20000
batch_size = 3
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
sess = tf.InteractiveSession()

for i in range(5):
    word = sess.run(next_element)
    spk_word = sess.run(next_spk)
    print(word['speaker'])
    print(spk_word['phoneList'])

