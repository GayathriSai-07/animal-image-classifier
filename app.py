import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="Animal Image Classifier",
    page_icon="🐾",
    layout="centered"
)

# ---------------------------------------
# Title
# ---------------------------------------

st.title("🐾 Animal Image Classifier")

st.write(
    "Upload an animal image and the AI will predict its class."
)

# ---------------------------------------
# Animal Classes
# ---------------------------------------

class_names = [
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

# ---------------------------------------
# Load Trained Model
# ---------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "animal_classifier.keras"
    )


try:
    model = load_model()

except Exception as e:

    st.error("Unable to load the AI model.")
    st.error(str(e))
    st.stop()


# ---------------------------------------
# Upload Image
# ---------------------------------------

uploaded_file = st.file_uploader(
    "Upload an animal image",
    type=["jpg", "jpeg", "png"]
)


# ---------------------------------------
# Image Prediction
# ---------------------------------------

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # -----------------------------------
    # Display Uploaded Image
    # -----------------------------------

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # -----------------------------------
    # Resize Image Internally
    # -----------------------------------
    # 224 x 224 is used internally.
    # It will NOT be displayed on the page.

    resized_image = image.resize(
        (224, 224)
    )

    # Convert image to NumPy array
    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # -----------------------------------
    # Make Prediction
    # -----------------------------------

    prediction = model.predict(
        image_array,
        verbose=0
    )

    # Find predicted class
    predicted_index = np.argmax(
        prediction[0]
    )

    predicted_animal = class_names[
        predicted_index
    ]

    # Calculate confidence
    confidence = (
        prediction[0][predicted_index] * 100
    )

    # -----------------------------------
    # Display Only Prediction
    # -----------------------------------

    st.success(
        f"Predicted Animal: {predicted_animal}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

else:

    st.info(
        "Please upload an image to get a prediction."
    )


# ---------------------------------------
# Footer
# ---------------------------------------

st.markdown("---")

st.caption(
    "AI-Based Animal Image Classification "
    "using TensorFlow and MobileNetV2"
)