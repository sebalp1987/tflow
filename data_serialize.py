import tensorflow as tf
import numpy as np

data = np.array([1., 2., 3., 4., 5.])

# Convert to Binary Format (Faster)
# First need that structure must be specified (Example structure)

def npy_to_tfrecords(fname, data):
    writer = tf.io.TFRecordWriter(fname)
    feature = {}
    feature['data'] = tf.train.Feature(float_list=tf.train.FloatList(value=data)) # dict of the data
    example = tf.train.Example(features=tf.train.Features(feature=feature)) # data
    serialized = example.SerializeToString() # is a sequence of binary strings
    writer.write(serialized)
    writer.close()

npy_to_tfrecords('file.tfrecords', data)

# To Read It Back
dataset = tf.data.TFRecordDataset('file.tfrecords')
# It needs a parse function that decodes it back
def parse_function(example_proto):
    keys_to_features = {'data': tf.io.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)}
    parsed_features = tf.io.parse_single_example(serialized=example_proto, features=keys_to_features)
    return parsed_features['data']
dataset = dataset.map(parse_function)

print(dataset)

# Now the iterator
iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
# array is retrieved as one item
item = iterator.get_next()
print(item)
print(item.numpy())
print(item[2].numpy())

