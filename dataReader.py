# This file is used to create a new dataset of Librispeech with tensorflow's API 
# to make the tensorflow training process and I/O simultaneously.

# We pad zeros in the end the MFCC frames to align for the use of using RNN structure.

import numpy as np
import tensorflow as tf

lines = 11232429
fo = open("Librispeech/cmvned_feats.ark")
fans = open("Librispeech/all_prons")
writer = tf.python_io.TFRecordWriter('%s.tfrecord' %'test')

data = []
sentenceNum = ''

for j in range(lines):
    if j%10000 == 0:
        print("Process to line ", j)
    
    line = fans.readline().split(' ')
    length = int(line[2])
    wordNum = int(line[3])
    phoneNum = [int(i) for i in line[4:]]
    speaker = line[0].split('-')[1]

    features={}
    features['speaker'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[speaker.encode()]))
    features['word'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[wordNum]))
    features['phoneList'] = tf.train.Feature(int64_list = tf.train.Int64List(value=phoneNum))
    
    if not sentenceNum == line[0].split('-')[2]:
        sentenceNum = line[0].split('-')[2]
        fo.readline()

    MFCC_pad0 = np.zeros(400*39)  # MFCC frame is at most 400 for a single word
    
    each_word = []
    for i in range(length):
        line = fo.readline()
        bb_list = filter(lambda k: k not in ['\n',']',']\n'],line.split(' '))
        bb_list_array = [float(i) for i in bb_list if i]
        each_word.append( bb_list_array )
    each_word = np.array(each_word)

    np.put(MFCC_pad0, np.arange(length), each_word)
    features['matrix'] = tf.train.Feature(float_list = tf.train.FloatList(value=MFCC_pad0))
    features['matrix_shape'] = tf.train.Feature(int64_list = tf.train.Int64List(value=(400, 39)))

    tf_features = tf.train.Features(feature= features)
    tf_example = tf.train.Example(features = tf_features)
    tf_serialized = tf_example.SerializeToString()
    writer.write(tf_serialized)

writer.close()

