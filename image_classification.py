"""
AI-Based 10 Animal Image Classification Project

This project uses a MobileNetV2 transfer-learning
model trained to classify images into 10 animal classes.

Image Size:
224 x 224

Number of Classes:
10
"""

import tensorflow as tf


CLASS_NAMES = [
    "Bear",
    "Bird",
    "Cat",
    "Cow",
    "Dog",
    "Elephant",
    "Horse",
    "Lion",
    "Tiger",
    "Zebra"
]


def load_classifier():
    """
    Load the trained Keras model.
    """

    model = tf.keras.models.load_model(
        "animal_classifier.keras"
    )

    return model


def get_class_names():
    """
    Return the animal class names.
    """

    return CLASS_NAMES


if __name__ == "__main__":

    model = load_classifier()

    print(
        "Model loaded successfully!"
    )

    print(
        "Input shape:",
        model.input_shape
    )

    print(
        "Output shape:",
        model.output_shape
    )

    print(
        "Animal classes:"
    )

    for animal in CLASS_NAMES:
        print(
            "-",
            animal
        )