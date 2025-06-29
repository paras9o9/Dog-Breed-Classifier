# 🐶 Dog Breed Classifier

The **Dog Breed Classifier** is a deep learning project that uses a convolutional neural network (CNN) with **transfer learning and fine-tuning** to classify images into over 120 different dog breeds. Built with TensorFlow and Gradio, this model offers a clean and interactive interface for image-based inference.

## 📌 Project Overview

This project tackles a real-world computer vision challenge: **multi-class image classification** where many classes (breeds) look visually similar. It progresses in three structured phases:

1. **Custom CNN** – Designed from scratch to understand the baseline performance.
2. **Transfer Learning** – Leveraged pretrained MobileNetV2 for better feature extraction.
3. **Fine-Tuning** – Unfrozen upper layers of the base model to improve prediction accuracy further.

By the end, the model achieves solid accuracy, especially considering class imbalance and visual similarity between certain breeds.

---

## 💡 Key Features

- 🔍 Classifies 120+ dog breeds from uploaded images
- 🧠 Uses pretrained MobileNetV2 + fine-tuning
- 📊 Returns top-3 predictions with confidence scores
- ⚙️ Easy-to-use Gradio web UI
- 🧪 Built with reproducibility in mind

---

## 🧠 Tech Stack

| Tool/Library     | Purpose                          |
|------------------|----------------------------------|
| **TensorFlow**   | Model architecture and training  |
| **Keras**        | Transfer learning & fine-tuning  |
| **NumPy**        | Efficient numerical processing   |
| **Gradio**       | Web interface for inference      |
| **Pillow**       | Image handling and preprocessing |

---

## 🗂️ Repository Contents

```bash
Dog_Breed_Classifier/
├── app.py                    # Gradio app for inference
├── dog_breed_classifier.keras  # Saved fine-tuned model
├── requirements.txt          # Dependencies
├── README.md                 # Project overview
└── utils/
    └── class_names.py        # (Optional) List of dog breeds
