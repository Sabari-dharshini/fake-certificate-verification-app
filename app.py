import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Fake Certificate Detector", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "certificate_forgery_model.keras",
        compile=False
    )

model = load_model()
CLASS_NAMES = ["clean", "fake_qr", "forged_seal", "tampered_grades"]

def predict(image):
    image = image.resize((224, 224))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    return CLASS_NAMES[np.argmax(preds)], float(np.max(preds))

st.title("🎓 Fake Degree / Certificate Verification System")

uploaded = st.file_uploader(
    "Upload Certificate Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Certificate", use_column_width=True)

    if st.button("Verify Certificate"):
        label, conf = predict(img)
        st.subheader("Verification Result")
        st.write(f"Prediction: **{label.upper()}**")
        st.write(f"Confidence: **{conf:.2f}**")

        if label == "clean":
            st.success("Certificate appears authentic")
        else:
            st.error("Forgery detected")
