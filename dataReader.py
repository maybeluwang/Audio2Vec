import numpy
import tensorflow as tf

lines = 600000
fo = open("Librispeech/cmvned_feats.ark")
fans = open("Librispeech/all_prons")
writer = tf.python_io.TFRecordWriter('%s.tfrecord' %'test')

data = []
sentenceNum = ''

for j in range(lines):
    if j%10000 ==0:
        print(j)
    
    line = fans.readline().split(' ')
    length = int(line[2])
    wordNum = int(line[1])
    phoneNum = [int(i) for i in line[3:]]
    speaker = line[0].split('-')[0]

    features={}
    features['speaker'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[speaker.encode()]))
    features['word'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[wordNum]))
    features['phoneList'] = tf.train.Feature(int64_list = tf.train.Int64List(value=phoneNum))
    
    if not sentenceNum == line[0].split('-')[2]:
        sentenceNum = line[0].split('-')[2]
        fo.readline()

    each_word = []
    for i in range(length):
        line = fo.readline()
        bb_list = filter(lambda k: k not in ['\n',']',']\n'],line.split(' '))
        each_word.append( [float(i) for i in bb_list if i])
    each_word = numpy.array(each_word)
    
    features['matrix'] = tf.train.Feature(float_list = tf.train.FloatList(value=each_word.reshape(-1)))
    features['matrix_shape'] = tf.train.Feature(int64_list = tf.train.Int64List(value=each_word.shape))

    tf_features = tf.train.Features(feature= features)
    tf_example = tf.train.Example(features = tf_features)
    tf_serialized = tf_example.SerializeToString()
    writer.write(tf_serialized)

writer.close()

