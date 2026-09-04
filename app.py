import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Six Animal Classifier",
    page_icon="🐾",
    layout="centered"
)

# Title
st.title("Animal Image Recognition Systemr")
st.write(
    "Upload an image and the CNN model will predict "
    "which animal it belongs to."
)

# Animal classes
class_names = [
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse"
]

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "six_animal_classifier.keras"
    )

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload an animal image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize image to CIFAR-10 size
    image = image.resize((32, 32))

    # Convert to array
    image_array = np.array(image)

    # Normalize
    image_array = image_array.astype("float32") / 255.0

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100

    # Display result
    st.success(
        f"Predicted Animal: {class_names[predicted_class]}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    # Show probabilities
    st.subheader("Prediction Probabilities")

    for i, animal in enumerate(class_names):
        st.write(
            f"{animal}: {prediction[0][i] * 100:.2f}%"
        )
