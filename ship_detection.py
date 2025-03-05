import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf


PATH = "ship_images/"
IMAGE_SCALING = (3, 3)


def prepare_workflow():
    tf.compat.v1.disable_v2_behavior()
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def load_image(path):
    image = cv2.imread(path)
    image = image[::IMAGE_SCALING[0], ::IMAGE_SCALING[1]]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# Metrics and Loss Function


# Intersection over Union for Objects
def IoU(y_true, y_pred, tresh=1e-10):
    Intersection = tf.keras.backend.sum(y_true * y_pred, axis=[1, 2, 3])
    Union = tf.keras.backend.sum(y_true, axis=[1, 2, 3]) + tf.keras.backend.sum(y_pred, axis=[1, 2, 3]) - Intersection
    return tf.keras.backend.mean((Intersection + tresh) / (Union + tresh), axis=0)


# Intersection over Union for Background
def back_IoU(y_true, y_pred):
    return IoU(1-y_true, 1-y_pred)


# Loss function
def IoU_loss(in_gt, in_pred):
    #return 2 - back_IoU(in_gt, in_pred) - IoU(in_gt, in_pred)
    return 1 - IoU(in_gt, in_pred)


def build_model():
    # Building model
    inputs = tf.keras.Input((256, 256, 3))

    c1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
    c1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2)) (c1)

    c2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2)) (c2)

    c3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2)) (c3)

    c4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c4)

    u5 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c4)
    u5 = tf.keras.layers.concatenate([u5, c3])
    c5 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (u5)
    c5 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c5)

    u6 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = tf.keras.layers.concatenate([u6, c2])
    c6 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (u6)
    c6 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c6)

    u7 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = tf.keras.layers.concatenate([u7, c1], axis=3)
    c7 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (u7)
    c7 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c7)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (c7)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer='adam', loss=IoU_loss, metrics=[IoU, back_IoU])

    # Loading weights
    model.load_weights('weights/weights')

    return model


def detect(model, file):
    images = np.zeros((1, 256, 256, 3))
    images[0] = load_image(PATH + file) / 255.0
    predictions = model.predict(images)

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(images[0])
    plt.title("Image")
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(predictions[0])
    plt.title("Detection")
    plt.show()

    return fig


prepare_workflow()
model = build_model()
