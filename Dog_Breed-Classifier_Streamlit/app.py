import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

# Load model from Hugging Face Hub
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="paras9o9/dog-breed-classifier",
        filename="dog_breed_classifier.keras",
        repo_type="model"
    )
    return tf.keras.models.load_model(model_path)

model = load_model()

# Load class names (should match training order)
class_names = class_names = [
    'affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
    'american_staffordshire_terrier', 'appenzeller', 'australian_terrier', 'basenji',
    'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog',
    'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick',
    'border_collie', 'border_terrier', 'borzoi', 'boston_bull',
    'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard',
    'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan',
    'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber',
    'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont',
    'dhole', 'dingo', 'doberman', 'english_foxhound', 'english_setter',
    'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever',
    'french_bulldog', 'german_shepherd', 'german_short-haired_pointer',
    'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane',
    'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound',
    'irish_setter', 'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound',
    'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie',
    'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever',
    'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois',
    'maltese_dog', 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
    'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound',
    'norwich_terrier', 'old_english_sheepdog', 'otterhound', 'papillon',
    'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback',
    'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke',
    'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
    'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
    'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle',
    'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier',
    'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner',
    'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet',
    'wire-haired_fox_terrier', 'yorkshire_terrier'
]

# App title
st.title("🐶 Dog Breed Classifier")
st.write("Upload a dog image and predict the top 3 most likely breeds.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_data = Image.open(uploaded_file).convert("RGB")
    st.image(image_data, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = image_data.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)[0]
    top_indices = predictions.argsort()[-3:][::-1]
    st.subheader("Top Predictions:")
    for i in top_indices:
        st.write(f"✅ {class_names[i]} ({predictions[i]*100:.2f}%)")
