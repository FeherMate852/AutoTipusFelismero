import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Modell betöltése
model = YOLO('runs/detect/train/weights/best.pt')

st.title("Autótípus felismerő")
uploaded_file = st.file_uploader("Tölts fel egy autó képet", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    try:
        # Kép betöltése
        image = Image.open(uploaded_file)

        # Átmeneti fájlba mentés
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            image.save(temp.name)
            temp_path = temp.name

        # Predikció
        results = model.predict(source=temp_path)
        st.image(image, caption='Feltöltött kép', use_column_width=True)
        st.write("Eredmények:")
        for box in results[0].boxes:
            cls = box.cls.cpu().numpy().item()
            confidence = box.conf.cpu().numpy().item()
            st.write(f"Osztály: {model.names[int(cls)]}, Bizonyosság: {confidence:.2f}")

    except Exception as e:
        st.error(f"Hiba történt: {e}")