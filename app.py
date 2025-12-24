import os
import time
import threading
import logging
import traceback
from typing import Optional

from flask import Flask, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge

# (Opcional, recomendado si tu app móvil usa WebView o si luego haces panel web)
# pip install flask-cors
try:
    from flask_cors import CORS
    _HAS_CORS = True
except Exception:
    _HAS_CORS = False

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
APP_NAME = "traductor-api"
MODEL_ID = os.getenv("MODEL_ID", "americasnlp/mt5-base-es-quw")
MAX_TEXT_LEN = int(os.getenv("MAX_TEXT_LEN", "300"))          # máximo caracteres de "texto"
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "200"))        # max_length para generación
MAX_BODY_MB = int(os.getenv("MAX_BODY_MB", "1"))              # máximo body en MB (JSON)
WARMUP_ON_START = os.getenv("WARMUP_ON_START", "true").lower() in ("1", "true", "yes")

# Flask / Gunicorn
PORT = int(os.getenv("PORT", "5000"))

# -------------------------------------------------------------------
# App & Logging
# -------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_MB * 1024 * 1024  # limita payload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(APP_NAME)

if _HAS_CORS:
    # Si no lo necesitas, puedes borrar esto (no afecta Insomnia ni server-to-server).
    CORS(app)

# -------------------------------------------------------------------
# Estado del modelo
# -------------------------------------------------------------------
translator = None  # type: Optional[object]
translator_loading = False
translator_error = None  # type: Optional[str]
translator_loaded_at = None  # type: Optional[float]


ABECEDARIO_KICHWA = {
    "a", "ch", "e", "h", "i", "k", "l", "ll", "m", "n", "ñ", "p", "q", "r", "s", "sh", "t", "u", "w", "y"
}


def _now() -> float:
    return time.time()


def load_translator() -> object:
    """
    Carga el pipeline. Se importa transformers dentro para que el import del módulo sea más ligero.
    """
    from transformers import pipeline
    return pipeline("translation", model=MODEL_ID)


def warmup_translator_async() -> None:
    """
    Carga el traductor en un thread para no bloquear requests.
    """
    global translator, translator_loading, translator_error, translator_loaded_at

    if translator_loading or translator is not None:
        return

    translator_loading = True
    translator_error = None

    def _job():
        global translator, translator_loading, translator_error, translator_loaded_at
        t0 = _now()
        try:
            logger.info(f"[WARMUP] Cargando modelo: {MODEL_ID}")
            translator = load_translator()
            translator_loaded_at = _now()
            logger.info(f"[WARMUP] Modelo cargado OK en {translator_loaded_at - t0:.2f}s")
        except Exception as e:
            translator = None
            translator_error = f"{e}"
            logger.error(f"[WARMUP] Error al cargar modelo: {e}\n{traceback.format_exc()}")
        finally:
            translator_loading = False

    threading.Thread(target=_job, daemon=True).start()


# -------------------------------------------------------------------
# Errores globales
# -------------------------------------------------------------------
@app.errorhandler(RequestEntityTooLarge)
def handle_large_body(_e):
    return jsonify({"error": f"Payload demasiado grande. Máximo {MAX_BODY_MB}MB."}), 413


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Ruta no encontrada"}), 404


@app.errorhandler(405)
def method_not_allowed(_e):
    return jsonify({"error": "Método no permitido"}), 405


@app.errorhandler(Exception)
def unhandled_exception(e):
    logger.critical(f"Error inesperado no manejado: {e}\n{traceback.format_exc()}")
    return jsonify({"error": "Error inesperado en el servidor"}), 500


# -------------------------------------------------------------------
# Health & Info
# -------------------------------------------------------------------
@app.get("/")
def root():
    return jsonify({
        "ok": True,
        "service": APP_NAME,
        "endpoints": ["/health", "/traducir"]
    })


@app.get("/health")
def health():
    """
    Útil para monitoreo (Render/uptime robot).
    No obliga a cargar el modelo, solo reporta estado.
    """
    status = "ready" if translator is not None else ("loading" if translator_loading else "not_ready")
    return jsonify({
        "ok": True,
        "service": APP_NAME,
        "model": MODEL_ID,
        "model_status": status,
        "model_error": translator_error,
        "model_loaded_at": translator_loaded_at
    })


# -------------------------------------------------------------------
# API principal
# -------------------------------------------------------------------
@app.post("/traducir")
def traducir():
    # 1) Validación de JSON
    if not request.data:
        return jsonify({"error": "No se envió información"}), 400

    try:
        data = request.get_json(force=True, silent=False)
    except Exception as json_error:
        logger.error(f"JSON inválido: {json_error}\n{traceback.format_exc()}")
        return jsonify({"error": "Formato JSON inválido"}), 400

    texto_es = data.get("texto", "")
    if not isinstance(texto_es, str):
        return jsonify({"error": "El campo 'texto' debe ser string"}), 400

    texto_es = texto_es.strip().lower()

    if not texto_es:
        return jsonify({"error": "No se envió texto"}), 400

    if len(texto_es) > MAX_TEXT_LEN:
        return jsonify({"error": f"Texto demasiado largo. Máximo {MAX_TEXT_LEN} caracteres."}), 400

    # 2) Abecedario: respuesta inmediata
    if texto_es in ABECEDARIO_KICHWA:
        return jsonify({"texto_es": texto_es, "traduccion": texto_es}), 200

    # 3) Si el modelo no está listo, no bloqueamos la request:
    #    devolvemos 503 y el cliente reintenta en unos segundos.
    if translator is None:
        # inicia warmup si no se hizo
        warmup_translator_async()

        if translator_loading:
            return jsonify({
                "error": "Modelo cargando. Intenta nuevamente en 20-60 segundos.",
                "status": "loading"
            }), 503

        # si no está cargando y sigue None, es fallo real
        return jsonify({
            "error": "El modelo de traducción no está disponible",
            "detail": translator_error
        }), 500

    # 4) Traducción
    t0 = _now()
    try:
        result = translator(texto_es, max_length=MAX_MODEL_LEN)
        traduccion = result[0].get("translation_text", "").strip()
        ms = int((_now() - t0) * 1000)

        logger.info(f"OK traducir ({ms}ms): '{texto_es}' -> '{traduccion}'")

        return jsonify({
            "texto_es": texto_es,
            "traduccion": traduccion,
            "ms": ms
        }), 200

    except Exception as model_error:
        logger.error(f"Error traducción: {model_error}\n{traceback.format_exc()}")
        return jsonify({"error": "Error interno de traducción"}), 500


# -------------------------------------------------------------------
# Arranque: opcional warmup al iniciar el worker
# -------------------------------------------------------------------
if WARMUP_ON_START:
    # en gunicorn puede ejecutarse por worker; igual está bien para evitar primera request lenta
    warmup_translator_async()


if __name__ == "__main__":
    logger.info(f"Iniciando {APP_NAME} en puerto {PORT} (model={MODEL_ID})")
    app.run(host="0.0.0.0", port=PORT)
