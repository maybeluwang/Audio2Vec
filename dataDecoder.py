import tensorflow as tf


filename = 'test.tfrecord'
dataset = tf.data.TFRecordDataset(filename)

#reader = tf.TFRecordReader() 
#filename_queue = tf.train.string_input_producer(
#            [filename], num_epochs=1
#        )
#_, serialized_example = reader.read(filename_queue) 

# by tensorflow API, read data parallel, fast, and easily
# only run in python >= 3.4
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


new_dataset = dataset.map(parse_function)
#new_dataset = new_dataset.padded_batch(4, padded_shapes=[None])

#new_dataset = new_dataset.batch(32)


# Create a function that can easily call next batch for the implementation 
# call by iteration
iterator = new_dataset.make_one_shot_iterator()
next_element = iterator.get_next()
sess = tf.InteractiveSession()
word = sess.run(next_element)
print(word)

'''
with tf.Session() as sess:
    init_op = tf.global_variables_initializer() 
    sess.run(init_op) 
    coord=tf.train.Coordinator() 
    threads= tf.train.start_queue_runners(coord=coord)
    example, l = sess.run([word,matrix])
'''
