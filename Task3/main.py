# Some of the functios are from: https://keras.io/examples/vision/siamese_network/

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet




BATCH_SIZE = 10000
IMG_SIZE = (200, 200)


"""train_dataset = tf.keras.utils.image_dataset_from_directory("Task3",
                                                            shuffle='true',
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)"""


def preprocess_image(filename):
    """
    Load the specified file as a JPG image, preprocess it and
    resize it to the target shape.
    """

    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMG_SIZE)
    return image


def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """
    print(anchor)
    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )

# dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
# create array from train triplets


fileObj = open("/Users/simonschindler/PycharmProjects/IML-main/Task3/train_triplets.txt", "r")  # opens the file in read mode
words = fileObj.read().splitlines()  # puts the file into an array
fileObj.close()
data = np.empty((len(words), 3), dtype=object)


print(words)
for i in range(len(words)):
    # temp = list(map(int, words[i].split()))
    dir = "/Users/simonschindler/PycharmProjects/IML-main/Task3/food/"

    temp = words[i].split()
    data[i][0] = dir + temp[0] + ".jpg"
    data[i][1] = dir + temp[1] + ".jpg"
    data[i][2] = dir + temp[2] + ".jpg"

    #tempArray = [dir + temp[0], dir + temp[1], dir + temp[2]]
    #data.append(tempArray)

print(data)

print(data[:, 0])
print(data[:, 1])
print(data[:, 2])
anchor_dataset = tf.data.Dataset.from_tensor_slices(data[:, 0])
positive_dataset = tf.data.Dataset.from_tensor_slices(data[:, 1])
negative_dataset = tf.data.Dataset.from_tensor_slices(data[:, 2])


dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.map(preprocess_triplets)

print(dataset)

train_dataset = dataset.take(round(len(words) * 0.8))
val_dataset = dataset.skip(round(len(words) * 0.8))
print(train_dataset)

train_dataset = train_dataset.batch(32, drop_remainder=False)
train_dataset = train_dataset.prefetch(8)

print(train_dataset)

val_dataset = val_dataset.batch(32, drop_remainder=False)
val_dataset = val_dataset.prefetch(8)

print(val_dataset)


"""def visualize(anchor, positive, negative):
    Visualize a few triplets from the supplied batches.

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])
    plt.show()


visualize(*list(train_dataset.take(1).as_numpy_iterator())[0])"""




base_cnn = resnet.ResNet50(
    weights="imagenet", input_shape=IMG_SIZE + (3,), include_top=False
)

flatten = layers.Flatten()(base_cnn.output)
dense1 = layers.Dense(512, activation="relu")(flatten)
dense1 = layers.BatchNormalization()(dense1)
dense2 = layers.Dense(256, activation="relu")(dense1)
dense2 = layers.BatchNormalization()(dense2)
output = layers.Dense(256)(dense2)

embedding = tf.keras.Model(base_cnn.input, output, name="Embedding")

trainable = False
for layer in base_cnn.layers:
    if layer.name == "conv5_block1_out":
        trainable = True
    layer.trainable = trainable

class DistanceLayer(layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


anchor_input = layers.Input(name="anchor", shape=IMG_SIZE + (3,))
positive_input = layers.Input(name="positive", shape=IMG_SIZE + (3,))
negative_input = layers.Input(name="negative", shape=IMG_SIZE + (3,))

distances = DistanceLayer()(
    embedding(resnet.preprocess_input(anchor_input)),
    embedding(resnet.preprocess_input(positive_input)),
    embedding(resnet.preprocess_input(negative_input)),
)

siamese_network = tf.keras.Model(
    inputs=[anchor_input, positive_input, negative_input], outputs=distances
)


class SiameseModel(tf.keras.Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]

checkpoint_path = "/Users/simonschindler/PycharmProjects/IML-main/Task3/cp"

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=False,
                                                 verbose=1, save_best_only=True)


siamese_model = SiameseModel(siamese_network)
siamese_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001))
siamese_model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=[cp_callback])

siamese_model.save('/Users/simonschindler/PycharmProjects/IML-main/Task3/models/my_model')



sample = next(iter(train_dataset))
#visualize(*sample)

anchor, positive, negative = sample
anchor_embedding, positive_embedding, negative_embedding = (
    embedding(resnet.preprocess_input(anchor)),
    embedding(resnet.preprocess_input(positive)),
    embedding(resnet.preprocess_input(negative)),
)


cosine_similarity = tf.keras.metrics.CosineSimilarity()

positive_similarity = cosine_similarity(anchor_embedding, positive_embedding)
print("Positive similarity:", positive_similarity.numpy())

negative_similarity = cosine_similarity(anchor_embedding, negative_embedding)
print("Negative similarity", negative_similarity.numpy())


# write results
