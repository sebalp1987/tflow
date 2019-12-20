import tensorflow as tf

filename = ['Book1.csv']
record_default = [tf.int32] * 2 # Two int columns
dataset = tf.data.experimental.CsvDataset(filename, record_default, header=True, field_delim=';')
print(dataset)
for item in dataset:
    print(item)