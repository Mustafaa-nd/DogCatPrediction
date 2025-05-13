import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Utilisation du GPU si dispo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔁 Charger le modèle fine-tuné
model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
model.load_state_dict(torch.load("mobilenetv2_finetuned.pt", map_location=device))
model.to(device)
model.eval()

# 🔄 ImageProcessor (nécessaire pour normalisation identique)
processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")

# Interface Streamlit
st.title("🐾 Prédiction d'Image : Chat ou Chien (Modèle Fine-tuné)")
st.write("Téléversez une image pour obtenir la prédiction.")

uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image téléversée")

    # Prétraitement avec les bons paramètres
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Prédiction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class_idx = logits.argmax(-1).item()
        confidence = probabilities[0, predicted_class_idx].item() * 100  # Pourcentage

    # 🔁 Classes selon ImageFolder : 0 → chat, 1 → chien
    id2label = {0: "Chat", 1: "Chien"}
    predicted_label = id2label[predicted_class_idx]

    st.success(f"✅ L’image est prédite comme : **{predicted_label}**")
    st.info(f"🔎 Confiance du modèle : **{confidence:.2f}%**")
