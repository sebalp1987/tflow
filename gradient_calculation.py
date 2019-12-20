import tensorflow as tf

w1 = tf.Variable(1.)
w2 = tf.Variable(1.)
w3 = tf.Variable(2.)

def weight_sum(x1, x2, x3):
    return w1*x1 + w2*x2 + w3*x3

with tf.GradientTape(persistent=True) as tape:
    sum = weight_sum(1., 2., 5.)
    print(sum)
    [w1_grad] = tape.gradient(sum, [w1])
    [w2_grad] = tape.gradient(sum, [w2])
    [w3_grad] = tape.gradient(sum, [w3])

# Permite calcular los X que hacen a la neurona

print(w1_grad.numpy())
print(w2_grad.numpy())
print(w3_grad.numpy())