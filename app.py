import os
import time
import threading
import logging
import traceback
from typing import Optional, Tuple

from flask import Flask, request, jsonify
from werkzeug.exceptions import RequestEntityTooLarge

try:
    from flask_cors import CORS
    _HAS_CORS = True
except Exception:
    _HAS_CORS = False


APP_NAME = "traductor-api"
MODEL_ID = os.getenv("MODEL_ID", "somosnlp-hackathon-2022/t5-small-finetuned-spanish-to-quechua")
MAX_TEXT_LEN = int(os.getenv("MAX_TEXT_LEN", "300"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "80"))
MAX_BODY_MB = int(os.getenv("MAX_BODY_MB", "1"))
WARMUP_ON_START = os.getenv("WARMUP_ON_START", "true").lower() in ("1", "true", "yes")
PORT = int(os.getenv("PORT", "5000"))

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

ABECEDARIO_KICHWA = {
    "a", "ch", "e", "h", "i", "k", "l", "ll", "m", "n", "ñ", "p", "q", "r", "s", "sh", "t", "u", "w", "y"
}


def _now() -> float:
    return time.time()


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(APP_NAME)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_MB * 1024 * 1024

if _HAS_CORS:
    if CORS_ORIGINS == "*" or not CORS_ORIGINS.strip():
        CORS(app)
    else:
        origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
        CORS(app, resources={r"/*": {"origins": origins}})


class ModelManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._loading = False
        self._error: Optional[str] = None
        self._loaded_at: Optional[float] = None
        self._bundle: Optional[Tuple[object, object]] = None

    @property
    def status(self) -> str:
        if self._bundle is not None:
            return "ready"
        if self._loading:
            return "loading"
        return "not_ready"

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def loaded_at(self) -> Optional[float]:
        return self._loaded_at

    def _load(self) -> Tuple[object, object]:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
        try:
            mdl.eval()
        except Exception:
            pass
        return tok, mdl

    def warmup_async(self) -> None:
        with self._lock:
            if self._loading or self._bundle is not None:
                return
            self._loading = True
            self._error = None

        def _job():
            t0 = _now()
            try:
                logger.info(f"[WARMUP] Cargando modelo: {MODEL_ID}")
                bundle = self._load()
                with self._lock:
                    self._bundle = bundle
                    self._loaded_at = _now()
                    self._error = None
                logger.info(f"[WARMUP] Modelo listo en {self._loaded_at - t0:.2f}s")
            except Exception as e:
                err = f"{e}"
                logger.error(f"[WARMUP] Error al cargar modelo: {err}\n{traceback.format_exc()}")
                with self._lock:
                    self._bundle = None
                    self._error = err
                    self._loaded_at = None
            finally:
                with self._lock:
                    self._loading = False

        threading.Thread(target=_job, daemon=True).start()

    def get_bundle_or_start(self) -> Optional[Tuple[object, object]]:
        with self._lock:
            bundle = self._bundle
            loading = self._loading

        if bundle is not None:
            return bundle

        if not loading:
            self.warmup_async()

        return None


model_mgr = ModelManager()


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
    logger.critical(f"Error no manejado: {e}\n{traceback.format_exc()}")
    return jsonify({"error": "Error inesperado en el servidor"}), 500


@app.get("/")
def root():
    return jsonify({"ok": True, "service": APP_NAME, "endpoints": ["/health", "/traducir"]})


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "service": APP_NAME,
            "model": MODEL_ID,
            "model_status": model_mgr.status,
            "model_error": model_mgr.error,
            "model_loaded_at": model_mgr.loaded_at,
        }
    )


@app.post("/traducir")
def traducir():
    if not request.data:
        return jsonify({"error": "No se envió información"}), 400

    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
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
        return jsonify({"texto_es": texto_es, "traduccion": texto_es, "ms": 0}), 200

    bundle = model_mgr.get_bundle_or_start()
    if bundle is None:
        if model_mgr.status == "loading":
            return jsonify({"error": "Modelo cargando. Intenta nuevamente en 10-60 segundos.", "status": "loading"}), 503
        return jsonify({"error": "El modelo no está disponible", "detail": model_mgr.error}), 500

    tok, mdl = bundle
    t0 = _now()
    try:
        inputs = tok(texto_es, return_tensors="pt", truncation=True)
        import torch

        with torch.no_grad():
            outputs = mdl.generate(
                inputs["input_ids"],
                max_length=MAX_MODEL_LEN,
                num_beams=4,
                early_stopping=True,
            )
        traduccion = tok.decode(outputs[0], skip_special_tokens=True).strip()
        ms = int((_now() - t0) * 1000)
        return jsonify({"texto_es": texto_es, "traduccion": traduccion, "ms": ms}), 200
    except Exception as e:
        logger.error(f"Error traducción: {e}\n{traceback.format_exc()}")
        return jsonify({"error": "Error interno de traducción"}), 500


@app.before_first_request
def _warmup_once():
    if WARMUP_ON_START:
        model_mgr.warmup_async()



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
