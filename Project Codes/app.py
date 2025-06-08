import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from PIL import Image

# ——— Custom layer definitions ———

# 1) FixedDropout (must match exactly what you used when training)
class FixedDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(FixedDropout, self).__init__(rate, **kwargs)

# 2) Ensure swish is registered
from tensorflow.keras.activations import swish

# ——— Load your model with all custom objects ———

MODEL_PATH = "mango_disease_classifier_enhanced.h5"

model = load_model(
    MODEL_PATH,
    custom_objects={
        "FixedDropout": FixedDropout,
        "swish": swish
    }
)

# ——— Disease classes & solutions ———

class_names = [
    "Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back",
    "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"
]
disease_solutions = {
    "Anthracnose": "Use copper-based or sulfur fungicides; ensure good airflow.",
    "Bacterial Canker": "Prune infected areas and apply copper bactericides.",
    "Cutting Weevil": "Apply insecticides targeting weevil larvae; keep area clean.",
    "Die Back": "Improve soil drainage, prune dead branches, and apply fungicides.",
    "Gall Midge": "Use neem oil or systemic insecticides early in the season.",
    "Healthy": "No treatment needed—you’ve got a healthy leaf!",
    "Powdery Mildew": "Treat with sulfur or potassium bicarbonate fungicides.",
    "Sooty Mould": "Control aphid/scale insects and wash leaves with mild soap."
}

# ——— Inference helper ———

def prepare_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, 0)

def predict_image(img_path):
    batch = prepare_image(img_path)
    preds = model.predict(batch)[0]
    idx = int(np.argmax(preds))
    disease = class_names[idx]
    solution = disease_solutions[disease]
    return f"**Prediction:** {disease}\n\n💡 **Solution:** {solution}"

# ——— Gradio interface ———

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Markdown(),
    title="🍃 Mango Leaf Disease Detection",
    description="Upload a mango leaf image; the model predicts the disease and suggests a solution.",
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch()
