import tensorflow as tf
import numpy as np

filename='file2.tfrecords'
data = {
    'ID': 61553,
    'Name': ['Jorge', 'Pedro'],
    'Scores': [62., 73.5]

}

ID = tf.train.Feature(int64_list=tf.train.Int64List(value=[data['ID']]))
Name = tf.train.Feature(bytes_list=tf.train.BytesList(value=[n.encode('utf-8') for n in data['Name']]))
Scores = tf.train.Feature(float_list=tf.train.FloatList(value=data['Scores']))

example = tf.train.Example(features=tf.train.Features(feature={'ID': ID, 'Name': Name, 'Scores': Scores}))

# Write
writer = tf.io.TFRecordWriter(filename)
writer.write(example.SerializeToString())
writer.close()

# Read Back
dataset = tf.data.TFRecordDataset(filename)

def parse_function(example_proto):
    keys_to_feature = {
        'ID': tf.io.FixedLenFeature([], dtype=tf.int64),
        'Name': tf.io.VarLenFeature(dtype=tf.string),
        'Scores': tf.io.VarLenFeature(dtype=tf.float32)
    }
    parsed_features = tf.io.parse_single_example(serialized=example_proto, features=keys_to_feature)
    return parsed_features['ID'], parsed_features['Name'], parsed_features['Scores']

dataset = dataset.map(parse_function)

# iterator

iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
item = iterator.get_next()
print(item)
print("ID: ",item[0].numpy())
name = item[1].values.numpy()
name1= name[0].decode()
name2 = name[1].decode('utf8')
print("Name:",name1,",",name2)
print("Scores: ",item[2].values.numpy())