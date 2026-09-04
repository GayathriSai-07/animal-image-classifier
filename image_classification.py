import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load the trained model
# --------------------------------------------------

model = tf.keras.models.load_model("six_animal_classifier.keras")

print("Model loaded successfully!")


# --------------------------------------------------
# 2. Define animal classes
# --------------------------------------------------

class_names = [
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse"
]


# --------------------------------------------------
# 3. Load CIFAR-10 test dataset
# --------------------------------------------------

(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Test dataset loaded successfully!")


# --------------------------------------------------
# 4. Select six animal classes
# --------------------------------------------------

selected_classes = [2, 3, 4, 5, 6, 7]

test_mask = np.isin(
    y_test.flatten(),
    selected_classes
)

x_test_animals = x_test[test_mask]
y_test_animals = y_test[test_mask].flatten()


# --------------------------------------------------
# 5. Convert original labels to 0-5
# --------------------------------------------------

label_mapping = {
    2: 0,   # Bird
    3: 1,   # Cat
    4: 2,   # Deer
    5: 3,   # Dog
    6: 4,   # Frog
    7: 5    # Horse
}

y_test_animals = np.array([
    label_mapping[label]
    for label in y_test_animals
])


# --------------------------------------------------
# 6. Normalize images
# --------------------------------------------------

x_test_animals = (
    x_test_animals.astype("float32") / 255.0
)


# --------------------------------------------------
# 7. Evaluate the trained model
# --------------------------------------------------

loss, accuracy = model.evaluate(
    x_test_animals,
    y_test_animals,
    verbose=1
)

print("\n--------------------------------")
print("Model Evaluation")
print("--------------------------------")

print(
    "Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Test Loss:",
    round(loss, 4)
)


# --------------------------------------------------
# 8. Predict one image
# --------------------------------------------------

index = 20

image = x_test_animals[index]

actual_label = y_test_animals[index]


prediction = model.predict(
    np.expand_dims(image, axis=0),
    verbose=0
)

predicted_label = np.argmax(prediction)


# --------------------------------------------------
# 9. Display prediction
# --------------------------------------------------

print("\n--------------------------------")
print("Image Prediction")
print("--------------------------------")

print(
    "Actual Animal:",
    class_names[actual_label]
)

print(
    "Predicted Animal:",
    class_names[predicted_label]
)


# --------------------------------------------------
# 10. Display the image
# --------------------------------------------------

plt.figure(figsize=(5, 5))

plt.imshow(image)

plt.title(
    "Actual: "
    + class_names[actual_label]
    + "\nPredicted: "
    + class_names[predicted_label]
)

plt.axis("off")

plt.show()


# --------------------------------------------------
# 11. Show prediction probabilities
# --------------------------------------------------

probabilities = prediction[0]

print("\n--------------------------------")
print("Prediction Probabilities")
print("--------------------------------")

for i in range(len(class_names)):

    print(
        class_names[i],
        ":",
        round(probabilities[i] * 100, 2),
        "%"
    )