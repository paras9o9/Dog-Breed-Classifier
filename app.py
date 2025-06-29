import tensorflow as tf
import numpy as np
import gradio as gr
from tensorflow.keras.preprocessing import image

# Load the model
model = tf.keras.models.load_model("dog_breed_classifier.keras")


# Class names (update this list if needed)
class_names = [
    # 👇 Replace this with your actual class list:
    "affenpinscher", "afghan_hound", "african_hunting_dog", "airedale",
    # ... all breed names in order from your training generator
]

def predict_breed(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions)
    confidence = float(np.max(predictions))
    return f"{class_names[train_gen.class_indices.keys()]} ({confidence * 100:.2f}%)"

gr.Interface(
    fn=predict_breed,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🐶 Dog Breed Classifier",
    description="Upload a dog photo and get its predicted breed!"
).launch()
