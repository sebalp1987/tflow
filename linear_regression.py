import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot

n_sample = 1000
training_steps = 1000
display_step = 100
lr = 0.01
m, c = 6, -5  # pendiente y constante como para empezar


def train_data(n, m, c):
    X = tf.random.normal([n])  # creamos una random de n puntos
    noise = tf.random.normal([n])  # Agregamos un poco de noise
    y = m * X + c + noise
    return X, y


def prediction(X, m, c):
    return X * m + c  # O X*weight + bias


def loss(x, y, weight, bias):
    error = prediction(x, weight, bias) - y
    sq_error = tf.square(error)
    return tf.reduce_mean(input_tensor=sq_error)


# Ahora podemos calcular la direccion y el valor de weight and biases (para hacer el Gradient Descendent)
def grad(x, y, weight, bias):
    with tf.GradientTape() as tape:
        loss_ = loss(x, y, weight, bias)
    return tape.gradient(loss_, [weight, bias])


# MODELO
x, y = train_data(n_sample, m, c)
plot.scatter(x, y)
plot.show()

# Inicializamos bias, weight (cosntante y pendiente)
W = tf.Variable(np.random.randn())  # Crea valores aleatorios de una distribucion normal standard
B = tf.Variable(np.random.randn())

# Probamos el loss inicial
print(loss(x, y, W, B))

# Training Loop
for step in range(training_steps):
    deltaW, deltaB = grad(x, y, W, B)  # Va mostrando los delta para donde nos debemos mover
    changeW = deltaW * lr
    changeB = deltaB * lr
    W.assign_sub(
        changeW)  # Las variables pueden ser cambiadas con esta funcion, reemplazamos entonces al nuevo valor W = W-changeW
    B.assign_sub(changeB)
    if step == 0 or step % display_step == 0:
        print(deltaB, deltaW)
        print("Loss at step {:02d}: {:.6f}".format(step, loss(x, y, W, B)))

# Final Loss
print("Final loss: {:.3f}".format(loss(x, y, W, B)))
print("W = {}, B = {}".format(W.numpy(), B.numpy()))
print("Compared with m = {:.3f}, c = {:.3f}".format(m, c)," of the original line")