import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
from tensorflow.python.keras.datasets import fashion_mnist
from keras.callbacks import ModelCheckpoint


batch_size = 128
epochs = 20
n_classes = 10
lr = 0.1
width = 28
height = 28

fashion_labels = ["Shirt/top", "Trousers", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                  "Ankle boot"]

# Load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
print(x_train.shape)
print(x_test.shape)

# Reshape
x_train = x_train.reshape((len(x_train), width * height))
x_test = x_test.reshape((len(x_test), width * height))
print(x_train.shape)
print(x_test.shape)

# Validation set
split = 50000
(x_train, x_valid) = x_train[:split], x_train[split:]
(y_train, y_valid) = y_train[:split], y_train[split:]

# One hot encode Y
y_train_ohe = tf.one_hot(y_train, depth=n_classes).numpy()
y_valid_ohe = tf.one_hot(y_valid, depth=n_classes).numpy()
y_test_ohe = tf.one_hot(y_test, depth=n_classes).numpy()

_, image = plot.subplots(1, 10, figsize=(8, 1))
for i in range(10):
    image[i].imshow(np.reshape(x_train[i], (width, height)), cmap="Greys")
    print(fashion_labels[y_train[i]], sep='', end='')

# Model
class LogisticRegression(tf.keras.Model): # Recomenado por Google que hedere siempre de Keras asi tenemos todas las funcionalidades
    def __init__(self, num_classes):
        super(LogisticRegression, self).__init__() # Lllama el constructor de tf.Model (es decir, todos sus atributos)
        self.dense = tf.keras.layers.Dense(num_classes) # Cuando instanciamos ya crea una red con un single layer vacio con 10 elementos.

    def call(self, inputs, training=None, mask=None): # Este es, que hace el output en cada forward
        output = self.dense(inputs) # toma el output de la variable self.dense (que es el Dense layer del init) y con eso lo pasara a un softmax abajo.
        with tf.device('/cpu:0'): # como softmax no existe en GPU forza a usar CPU
            output = tf.nn.softmax(output)

        return output

model = LogisticRegression(n_classes)
optimiser = tf.keras.optimizers.Adam(lr)
model.compile(optimiser=optimiser, loss='categorical_crossentropy', metrics=['accuracy'])

# Keras usa todo el dataset para saber el shape, asi que hacemos un dummy call para que ya sepa mas facil
dummy_x = tf.zeros((1, height*width))
model.call(dummy_x)

# Checkpoint para salvar el mejor modelo
checkpoint = ModelCheckpoint('best.hdf5', verbose=2, save_best_only=True, save_weights_only=True)

# Train
model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=epochs, validation_data=(x_valid, y_valid_ohe),
          callbacks=[checkpoint], verbose=2)
model.load_weights('best.hdf5')

# Test
scores = model.evaluate(x_test, y_test_ohe, batch_size, verbose=2)
print('Accuracy ', scores)

# Predict
y_predictions = model.predict(x_test)

# Check a random sample of predictions
size = 12 # images
rows = 3
cols = 4
index = 42

fig = plot.figure(figsize=(15, 3))
for i, index in enumerate(np.random.choice(x_test.shape[0], size=size, replace=False)):
    axis = fig.add_subplot(rows, cols, i + 1, xticks=[],
                           yticks=[])  # position i+1 in grid with rows rows and cols columns
    axis.imshow(x_test[index].reshape(width, height), cmap="Greys")
    index_predicted = np.argmax(y_predictions[index])
    index_true = np.argmax(y_test_ohe[index])
    axis.set_title(("{} ({})").format(fashion_labels[index_predicted], fashion_labels[index_true]),
                   color=("green" if index_predicted == index_true else "red"))

plot.show()

