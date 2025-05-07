from transformers import AutoTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np

# Load model dan tokenizer
model = TFBertForSequenceClassification.from_pretrained('news_classification')
tokenizer = AutoTokenizer.from_pretrained('news_classification')

# Tes input
texts = [
    "Ini berita hoax",
    "Ini berita bukan hoax",
]
for text in texts:
    inputs = tokenizer.encode_plus(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="tf"
    )
    predictions = model(inputs["input_ids"]).logits
    probabilities = tf.nn.softmax(predictions, axis=1).numpy()
    pred_class = np.argmax(probabilities, axis=1).item()

    print(f"Teks: {text}")
    print(f"Probabilitas: {probabilities}")
    print(f"Prediksi kelas: {pred_class}")
