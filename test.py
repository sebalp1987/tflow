import tensorflow as tf

t1 = tf.Variable([[[0., 1., 2.]], [[3., 4., 5.]]])
print(t1)
print(t1.shape)
print(tf.rank(t1))
print(t1[1, 0, 1].numpy())
print(t1.numpy())
print(tf.size(input=t1).numpy())
t2 = tf.constant(2.)
print(t1*t2)
print(t1*2.)

u = tf.constant([[1, 2, 3]])
v = tf.constant([[4, 5, 6]])
print(tf.matmul(u, tf.transpose(a=v)))

print(tf.reduce_mean(t1))
print(tf.reduce_mean(t1, axis=0))
print(tf.reduce_mean(t1, axis=1))

print(tf.random.normal(shape=(3,2 ), mean=10, stddev=1))

def f1(x, y):
    return tf.reduce_mean(input_tensor=tf.multiply(x, y))

funct = tf.function(f1)

x = tf.constant([2, 2])
y = tf.constant([3, 3])

print(f1(x, y))
print(funct(x, y))