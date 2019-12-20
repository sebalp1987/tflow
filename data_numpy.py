import tensorflow as tf
import numpy as np

num_items = 11
nlist1 = np.arange(num_items)
nlist2 = np.arange(num_items, num_items *2)

# PIPELINE
# Create Dataset
nlist1_dataset = tf.data.Dataset.from_tensor_slices(nlist1)
print(nlist1_dataset)

# Create Iterator to Access Data
iterator = tf.compat.v1.data.make_one_shot_iterator(nlist1_dataset) # One shot: I can only call it once
print(iterator)
for item in nlist1_dataset:
    num = iterator.get_next().numpy()
    print(num)

# Or Detaset By Batches
nlist1_dataset = tf.data.Dataset.from_tensor_slices(nlist1).batch(3, drop_remainder=True)
iterator = tf.compat.v1.data.make_one_shot_iterator(nlist1_dataset)
for item in nlist1_dataset:
    num = iterator.get_next().numpy()
    print(num)
