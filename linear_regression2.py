import tensorflow as tf
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
import matplotlib.pyplot as plot

lr = 0.01
epochs = 1000
display_epoch = epochs // 20
n_train = 300
n_valid = 100

features, prices = load_boston(True)
n_test = len(features) - n_train - n_valid

x_train = tf.cast(scale(features[:n_train]), dtype=tf.float32) # Escalamos N(0,1)
y_train = prices[:n_train]

x_valid = tf.cast(scale(features[n_train: n_train + n_valid]), dtype=tf.float32)
y_valid = prices[n_train: n_train + n_valid]

x_test = tf.cast(scale(features[n_train + n_valid: n_train + n_valid + n_test]), dtype=tf.float32)
y_test = prices[n_train + n_valid: n_train + n_valid + n_test]

def prediction(x, w, b):
    return tf.matmul(x, w) + b

def loss(x, y, w, b):
    pred = prediction(x, w, b)
    error = y - pred
    error_sq = tf.square(error)
    return tf.sqrt(tf.reduce_mean(input_tensor=error_sq))

def gradient(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_ = loss(x, y, w, b)
        return tape.gradient(loss_, [w, b])


W = tf.Variable(tf.random.normal([13, 1], mean=0., stddev= 1.0, dtype=tf.float32)) # Son 13 variables
B = tf.Variable(tf.zeros(1), dtype=tf.float32)

# Initial Loss
print(loss(x_train, y_train, W, B))

# Training Loop
for e in range(epochs):
    deltaW, deltaB = gradient(x_train, y_train, W, B)
    changeW = deltaW * lr
    changeB = deltaB * lr
    W.assign_sub(changeW)
    B.assign_sub(changeB)
    if e==0 or e % display_epoch == 0:
        print(deltaW.numpy(), deltaB.numpy())
        print("Validation loss after epoch {:02d}: {:.3f}".format(e, loss(x_valid, y_valid, W, B)))

print('final W, B', W.numpy(), B.numpy())

example_house = 69
y = y_test[example_house]
y_pred = prediction(x_test, W.numpy(),B.numpy())[example_house]
print("Actual median house value",y," in $10K")
print("Predicted median house value ",y_pred.numpy()," in $10K")

# Test Prediction
y_pred = prediction(x_test, W, B).numpy()
y_pred = np.reshape(y_pred, y_test.shape)
plot.scatter(y_pred, y_test)
plot.show()

print(y_pred)
print(y_test)