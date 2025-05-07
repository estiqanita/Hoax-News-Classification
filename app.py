import tensorflow as tf
from transformers import AutoTokenizer
import numpy as np
import streamlit as st
from transformers import TFBertForSequenceClassification
from keras.models import load_model
from keras.layers import Dense, Input



# Load model dan tokenizer dengan caching
@st.cache_resource
def load_model():
    return TFBertForSequenceClassification.from_pretrained('news_classification')

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained('news_classification')

model = load_model()
tokenizer = load_tokenizer()

# Tampilan aplikasi
st.markdown("""
    <div style='display: flex; align-items: center; gap: 15px;'>
        <img src='https://img.icons8.com/?size=100&id=1UIFaCg7Q3lR&format=png&color=000000' width='50'>
        <h1 style='margin: 0;'>Hoax News Prediction</h1>
    </div>
""", unsafe_allow_html=True)
st.write("Masukkan teks berita yang ingin diprediksi apakah hoax atau bukan.")

# Input teks
news_text = st.text_area("Masukkan teks berita di sini:", height=150)

# Tombol prediksi
if st.button("Prediksi"):
    if not news_text.strip():
        st.warning("Mohon masukkan teks berita terlebih dahulu.")
    elif len(news_text.split()) > 512:
        st.warning("Teks terlalu panjang, mohon masukkan maksimal 512 kata.")
    else:
        st.info("Sedang melakukan prediksi...")
        
        # Preprocessing teks
        inputs = tokenizer.encode_plus(
            news_text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="tf"
        )
        
        # Prediksi menggunakan model
        predictions = model(inputs["input_ids"]).logits  # Dapatkan logits dari model
        probabilities = tf.nn.softmax(predictions, axis=1).numpy()  # Konversi logits ke probabilitas
        pred_class = np.argmax(probabilities, axis=1).item()  # Ambil kelas dengan probabilitas tertinggi

        # Cek prediksi
        if pred_class == 0:
            pred = "Berita Bukan Hoax"
        else:
            pred = "Berita Hoax"
        
        # Tampilkan hasil
        st.success(f"Prediksi selesai! {pred}")
        st.write(f"Probabilitas:")
        st.write(f"- Berita Bukan Hoax: {probabilities[0][0]:.2%}")
        st.write(f"- Berita Hoax: {probabilities[0][1]:.2%}")


