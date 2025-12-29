import os
import time
import threading
import logging
import traceback
from typing import Optional

from flask import Flask, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge

try:
    from flask_cors import CORS
    _HAS_CORS = True
except Exception:
    _HAS_CORS = False

APP_NAME = "traductor-api"
MODEL_ID = os.getenv("MODEL_ID", "facebook/nllb-200-distilled-600M")
SRC_LANG = os.getenv("SRC_LANG", "spa_Latn")
TGT_LANG = os.getenv("TGT_LANG", "quy_Latn")
MAX_TEXT_LEN = int(os.getenv("MAX_TEXT_LEN", "300"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "200"))
MAX_BODY_MB = int(os.getenv("MAX_BODY_MB", "1"))
WARMUP_ON_START = os.getenv("WARMUP_ON_START", "true").lower() in ("1", "true", "yes")
PORT = int(os.getenv("PORT", "5000"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_MB * 1024 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(APP_NAME)

if _HAS_CORS:
    CORS(app)

translator = None  # type: Optional[object]
translator_loading = False
translator_error = None  # type: Optional[str]
translator_loaded_at = None  # type: Optional[float]

ABECEDARIO_KICHWA = {"a", "ch", "e", "h", "i", "k", "l", "ll", "m", "n", "ñ", "p", "q", "r", "s", "sh", "t", "u", "w", "y"}


def _now() -> float:
    return time.time()


def load_translator() -> object:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    tok.src_lang = SRC_LANG
    pipe = pipeline("translation", model=mdl, tokenizer=tok, device=-1)

    if not hasattr(tok, "lang_code_to_id") or TGT_LANG not in tok.lang_code_to_id:
        raise ValueError(f"TGT_LANG '{TGT_LANG}' no existe en este modelo. Verifica SRC_LANG/TGT_LANG.")
    pipe._forced_bos_token_id = tok.lang_code_to_id[TGT_LANG]
    return pipe


def warmup_translator_async() -> None:
    global translator, translator_loading, translator_error, translator_loaded_at

    if translator_loading or translator is not None:
        return

    translator_loading = True
    translator_error = None

    def _job():
        global translator, translator_loading, translator_error, translator_loaded_at
        t0 = _now()
        try:
            logger.info(f"[WARMUP] Cargando modelo: {MODEL_ID} | {SRC_LANG} -> {TGT_LANG}")
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


@app.get("/")
def root():
    return jsonify({"ok": True, "service": APP_NAME, "endpoints": ["/health", "/traducir"]})


@app.get("/health")
def health():
    status = "ready" if translator is not None else ("loading" if translator_loading else "not_ready")
    return jsonify(
        {
            "ok": True,
            "service": APP_NAME,
            "model": MODEL_ID,
            "src_lang": SRC_LANG,
            "tgt_lang": TGT_LANG,
            "model_status": status,
            "model_error": translator_error,
            "model_loaded_at": translator_loaded_at,
        }
    )


@app.post("/traducir")
def traducir():
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

    if texto_es in ABECEDARIO_KICHWA:
        return jsonify({"texto_es": texto_es, "traduccion": texto_es}), 200

    if translator is None:
        warmup_translator_async()

        if translator_loading:
            return jsonify({"error": "Modelo cargando. Intenta nuevamente en 20-60 segundos.", "status": "loading"}), 503

        return jsonify({"error": "El modelo de traducción no está disponible", "detail": translator_error}), 500

    t0 = _now()
    try:
        result = translator(texto_es, max_length=MAX_MODEL_LEN, forced_bos_token_id=translator._forced_bos_token_id)
        traduccion = result[0].get("translation_text", "").strip()
        ms = int((_now() - t0) * 1000)
        logger.info(f"OK traducir ({ms}ms): '{texto_es}' -> '{traduccion}'")
        return jsonify({"texto_es": texto_es, "traduccion": traduccion, "ms": ms}), 200
    except Exception as model_error:
        logger.error(f"Error traducción: {model_error}\n{traceback.format_exc()}")
        return jsonify({"error": "Error interno de traducción"}), 500


if WARMUP_ON_START:
    warmup_translator_async()


if __name__ == "__main__":
    logger.info(f"Iniciando {APP_NAME} en puerto {PORT} (model={MODEL_ID})")
    app.run(host="0.0.0.0", port=PORT)
