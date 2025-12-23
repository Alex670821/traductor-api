from flask import Flask, request, jsonify
from transformers import pipeline
import logging
import traceback

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

abecedario_kichwa = [
    "a","ch","e","h","i","k","l","ll","m","n","ñ","p","q","r","s","sh","t","u","w","y"
]

translator = None

def load_translator():
    try:
        t = pipeline("translation", model="americasnlp/mt5-base-es-quw")
        logging.info("Modelo de traducción Kichwa ecuatoriano cargado correctamente.")
        return t
    except Exception as e:
        logging.error(f"Error al cargar el modelo de traducción: {e}\n{traceback.format_exc()}")
        return None

def get_translator():
    global translator
    if translator is None:
        translator = load_translator()
    return translator

@app.route("/traducir", methods=["POST"])
def traducir():
    try:
        if not request.data:
            return jsonify({"error": "No se envió información"}), 400

        try:
            data = request.get_json(force=True)
        except Exception:
            return jsonify({"error": "Formato JSON inválido"}), 400

        texto_es = data.get("texto", "")
        if not isinstance(texto_es, str):
            return jsonify({"error": "El campo 'texto' debe ser string"}), 400

        texto_es = texto_es.strip().lower()
        if not texto_es:
            return jsonify({"error": "No se envió texto"}), 400

        if texto_es in abecedario_kichwa:
            return jsonify({"texto_es": texto_es, "traduccion": texto_es})

        t = get_translator()
        if t is None:
            return jsonify({"error": "El modelo de traducción no está disponible"}), 500

        traduccion = t(texto_es, max_length=200)[0]["translation_text"]
        return jsonify({"texto_es": texto_es, "traduccion": traduccion})

    except Exception as unexpected:
        logging.critical(f"Error inesperado: {unexpected}\n{traceback.format_exc()}")
        return jsonify({"error": "Error inesperado en el servidor"}), 500
