import joblib
import pandas as pd
from flask import Flask, request, jsonify

# ==============================
# 1. Inicializar la aplicación
# ==============================
app = Flask(__name__)

# ==============================
# 2. Cargar el modelo y vectorizador
# ==============================
modelo = joblib.load("modelo_sentimientos.pkl")
vectorizer = joblib.load("vectorizer_tfidf.pkl")

# ==============================
# 3. Ruta de prueba
# ==============================
@app.route('/')
def home():
    return jsonify({"mensaje": "✅ API de Análisis de Sentimientos funcionando correctamente"})

# ==============================
# 4. Endpoint de predicción
# ==============================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recibir JSON desde el cliente
        data = request.get_json()

        if not data or 'texto' not in data:
            return jsonify({"error": "Falta el campo 'texto' en el JSON"}), 400

        texto = data['texto']
        X = vectorizer.transform([texto])
        pred = modelo.predict(X)[0]
        sentimiento = "POSITIVO 😊" if pred == 1 else "NEGATIVO 😞"

        # Respuesta en formato JSON
        return jsonify({
            "texto": texto,
            "sentimiento": sentimiento,
            "valor": int(pred)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# 5. Ejecutar servidor
# ==============================
if __name__ == '__main__':
    app.run(debug=True, port=8000)
