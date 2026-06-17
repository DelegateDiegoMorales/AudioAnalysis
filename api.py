"""
api.py — API REST para análisis de audio con openSMILE
Deploy en Railway: recibe WAV, devuelve scores de nerviosismo/confianza/energía

ENDPOINT:
    POST /analizar
        Body: multipart/form-data, campo "audio" = archivo WAV
        Returns: JSON { scores, interpretacion, features_raw }

    GET /health
        Returns: { "status": "ok" }
"""

import os
import json
import tempfile
import numpy as np
import soundfile as sf
import opensmile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="MentorIA Audio Analyzer", version="1.0")

# Inicializar openSMILE una sola vez al arrancar (costoso, no por request)
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# ─────────────────────────────────────────────────────────────────
# Misma lógica de scoring que analizar_audio.py
# ─────────────────────────────────────────────────────────────────

RANGOS = {
    "F0semitoneFrom27.5Hz_sma3nz_amean":       (15.0, 30.0),
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm":  (0.1,  0.4),
    "jitterLocal_sma3nz_amean":                (0.002, 0.015),
    "shimmerLocaldB_sma3nz_amean":             (0.2,  0.8),
    "HNRdBACF_sma3nz_amean":                   (10.0, 20.0),
    "loudness_sma3_stddevNorm":                (0.2,  0.6),
    "spectralFlux_sma3_amean":                 (0.3,  1.0),
}

PESOS_NERVIOSISMO = {
    "jitterLocal_sma3nz_amean":                0.30,
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm":  0.20,
    "shimmerLocaldB_sma3nz_amean":             0.20,
    "HNRdBACF_sma3nz_amean":                   0.20,
    "loudness_sma3_stddevNorm":                0.10,
}


def normalizar(nombre, valor):
    if nombre not in RANGOS:
        return 0.5
    minv, maxv = RANGOS[nombre]
    rango = maxv - minv
    if rango == 0:
        return 0.5
    norm = (1.0 - (valor - minv) / rango) if "HNR" in nombre else (valor - minv) / rango
    return float(np.clip(norm, 0.0, 1.0))


def calcular_scores(fd):
    score_nerv = 0.0
    detalles   = {}
    for feat, peso in PESOS_NERVIOSISMO.items():
        if feat in fd:
            n = normalizar(feat, fd[feat])
            score_nerv += n * peso
            detalles[feat] = round(n, 3)

    nerviosismo = round(min(score_nerv * 10, 10.0), 2)

    pitch_est  = 1.0 - detalles.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", 0.5)
    hnr_alto   = 1.0 - detalles.get("HNRdBACF_sma3nz_amean", 0.5)
    jit_bajo   = 1.0 - detalles.get("jitterLocal_sma3nz_amean", 0.5)
    confianza  = round(min((pitch_est * 0.4 + hnr_alto * 0.35 + jit_bajo * 0.25) * 10, 10.0), 2)

    ln = float(np.clip((fd.get("loudness_sma3_amean", 0.3) - 0.1) / 0.8, 0.0, 1.0))
    fn = float(np.clip((fd.get("spectralFlux_sma3_amean", 0.5) - 0.1) / 1.5, 0.0, 1.0))
    energia = round(min((ln * 0.6 + fn * 0.4) * 10, 10.0), 2)

    return nerviosismo, confianza, energia


def interpretar(n, c, e):
    t_n = ("tranquilo y controlado" if n <= 3
           else "nerviosismo moderado" if n <= 6
           else "nerviosismo alto")
    t_c = ("voz poco confiada" if c <= 3
           else "confianza moderada" if c <= 6
           else "voz confiada y estable")
    t_e = ("tono apagado" if e <= 3
           else "energía moderada" if e <= 6
           else "voz expresiva y enérgica")
    return f"{t_n}, {t_c}, {t_e}."


# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analizar")
async def analizar(audio: UploadFile = File(...)):
    # Validar extensión
    if not audio.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos WAV.")

    # Guardar en archivo temporal
    contenido = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(contenido)
        ruta_tmp = tmp.name

    try:
        # Verificar duración mínima
        info = sf.info(ruta_tmp)
        duracion = info.frames / info.samplerate
        if duracion < 2.0:
            raise HTTPException(status_code=400, detail="Audio muy corto (mínimo 2 segundos).")

        # Extraer features
        df = smile.process_file(ruta_tmp)
        if df.empty:
            raise HTTPException(status_code=422, detail="No se pudieron extraer features del audio.")

        fd = {col: float(df[col].iloc[0]) for col in df.columns}

        # Calcular scores
        nerviosismo, confianza, energia = calcular_scores(fd)

        return JSONResponse({
            "scores": {
                "nerviosismo": nerviosismo,
                "confianza":   confianza,
                "energia":     energia,
            },
            "interpretacion": interpretar(nerviosismo, confianza, energia),
            "duracion_s":     round(duracion, 2),
            "features_raw": {
                "pitch_medio":         round(fd.get("F0semitoneFrom27.5Hz_sma3nz_amean", 0), 4),
                "pitch_variabilidad":  round(fd.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", 0), 4),
                "jitter":              round(fd.get("jitterLocal_sma3nz_amean", 0), 6),
                "shimmer_dB":          round(fd.get("shimmerLocaldB_sma3nz_amean", 0), 4),
                "HNR_dB":              round(fd.get("HNRdBACF_sma3nz_amean", 0), 4),
                "loudness_media":      round(fd.get("loudness_sma3_amean", 0), 4),
                "loudness_variacion":  round(fd.get("loudness_sma3_stddevNorm", 0), 4),
                "spectral_flux":       round(fd.get("spectralFlux_sma3_amean", 0), 4),
            }
        })

    finally:
        os.unlink(ruta_tmp)
