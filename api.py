"""
api.py — API REST para análisis de audio con openSMILE
Deploy en Render/Railway: recibe WAV, devuelve 7 scores prosódicos y emocionales.

ENDPOINTS:
    GET  /health     → {"status": "ok"}
    POST /analizar   → multipart WAV → JSON con scores de audio
"""

import os
import tempfile
import numpy as np
import soundfile as sf
import opensmile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="MentorIA Audio Analyzer", version="2.0")

# Inicializar openSMILE una sola vez al arrancar (costoso, no por request)
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# ─────────────────────────────────────────────────────────────────────────────
# Rangos de referencia (hablante adulto en conversación relajada)
# ─────────────────────────────────────────────────────────────────────────────

PESOS_NERVIOSISMO = {
    "jitterLocal_sma3nz_amean":               0.30,
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": 0.20,
    "shimmerLocaldB_sma3nz_amean":            0.20,
    "HNRdBACF_sma3nz_amean":                  0.20,   # invertido
    "loudness_sma3_stddevNorm":               0.10,
}

RANGOS = {
    "jitterLocal_sma3nz_amean":               (0.002, 0.015),
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": (0.10,  0.40),
    "shimmerLocaldB_sma3nz_amean":            (0.20,  0.80),
    "HNRdBACF_sma3nz_amean":                  (10.0,  20.0),
    "loudness_sma3_stddevNorm":               (0.20,  0.60),
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
    # ── Nerviosismo (0-10) ─────────────────────────────────────────────────
    score_nerv = 0.0
    det = {}
    for feat, peso in PESOS_NERVIOSISMO.items():
        if feat in fd:
            n = normalizar(feat, fd[feat])
            score_nerv += n * peso
            det[feat] = round(n, 3)

    nerviosismo = round(min(score_nerv * 10, 10.0), 2)

    # ── Confianza (0-10) ───────────────────────────────────────────────────
    pitch_est = 1.0 - det.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", 0.5)
    hnr_alto  = 1.0 - det.get("HNRdBACF_sma3nz_amean",                  0.5)
    jit_bajo  = 1.0 - det.get("jitterLocal_sma3nz_amean",               0.5)
    confianza = round(min((pitch_est * 0.4 + hnr_alto * 0.35 + jit_bajo * 0.25) * 10, 10.0), 2)

    # ── Energía / Entusiasmo (0-10) ────────────────────────────────────────
    ln      = float(np.clip((fd.get("loudness_sma3_amean",    0.3) - 0.1) / 0.8, 0.0, 1.0))
    fn      = float(np.clip((fd.get("spectralFlux_sma3_amean", 0.5) - 0.1) / 1.5, 0.0, 1.0))
    energia = round(min((ln * 0.6 + fn * 0.4) * 10, 10.0), 2)

    # ── Monotonía (0-10, 10 = muy monótono) ───────────────────────────────
    # f0_std bajo = pitch plano = monótono
    # Rango natural: ~0.05 (muy plano) → ~0.50 (muy expresivo)
    f0_std    = fd.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", 0.25)
    mono_n    = float(np.clip(1.0 - (f0_std / 0.50), 0.0, 1.0))
    monotonia = round(mono_n * 10, 2)

    # ── Dinamismo vocal (0-10) ─────────────────────────────────────────────
    # stddev del volumen + spectral flux como proxies de expresividad
    loud_std = fd.get("loudness_sma3_stddevNorm", 0.30)
    # spectralFlux stddev puede no existir; usar amean como proxy
    flux_val = fd.get("spectralFlux_sma3_stddevNorm",
                      fd.get("spectralFlux_sma3_amean", 0.50))
    d_loud    = float(np.clip((loud_std - 0.10) / 0.50, 0.0, 1.0))
    d_flux    = float(np.clip((flux_val  - 0.10) / 1.40, 0.0, 1.0))
    dinamismo = round((d_loud * 0.60 + d_flux * 0.40) * 10, 2)

    # ── Velocidad del habla (0-10, ~5 = normal) ───────────────────────────
    # Neutral si VoicedSegmentsPerSec no existe en el feature set
    vps = fd.get("VoicedSegmentsPerSec", None)
    velocidad = 5.0 if vps is None else round(float(np.clip(vps / 6.0, 0.0, 1.0)) * 10, 2)

    # ── Ratio de pausas (0-10, 10 = muchas pausas largas) ─────────────────
    # Neutral si MeanUnvoicedSegmentLength no existe
    mun = fd.get("MeanUnvoicedSegmentLength", None)
    ratio_pausas = 5.0 if mun is None else round(float(np.clip((mun - 0.05) / 0.60, 0.0, 1.0)) * 10, 2)

    return {
        "nerviosismo":  nerviosismo,
        "confianza":    confianza,
        "energia":      energia,
        "monotonia":    monotonia,
        "dinamismo":    dinamismo,
        "velocidad":    velocidad,
        "ratio_pausas": ratio_pausas,
    }


def interpretar(scores):
    n  = scores["nerviosismo"]
    c  = scores["confianza"]
    e  = scores["energia"]
    mo = scores["monotonia"]
    di = scores["dinamismo"]
    v  = scores["velocidad"]
    p  = scores["ratio_pausas"]

    partes = []

    partes.append("tranquilo y controlado" if n <= 3 else
                  "nerviosismo moderado"   if n <= 6 else
                  "nerviosismo alto")

    partes.append("voz confiada y estable" if c >= 7 else
                  "confianza moderada"     if c >= 4 else
                  "voz insegura")

    partes.append("alta energía"   if e >= 7 else
                  "energía moderada" if e >= 4 else
                  "tono apagado")

    if mo >= 7:
        partes.append("habla muy monótona")
    elif mo >= 4:
        partes.append("entonación algo plana")

    if di >= 7:
        partes.append("voz muy expresiva")
    elif di <= 3:
        partes.append("poca variación vocal")

    partes.append("habla lenta"      if v <= 3 else
                  "habla rápida"     if v >= 7 else
                  "velocidad normal")

    if p >= 7:
        partes.append("muchas pausas o dudas")
    elif p <= 3:
        partes.append("flujo continuo")

    return ", ".join(partes) + "."


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analizar")
async def analizar(audio: UploadFile = File(...)):
    if not audio.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos WAV.")

    contenido = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(contenido)
        ruta_tmp = tmp.name

    ruta_analisis = ruta_tmp   # puede cambiar si normalizamos

    try:
        info     = sf.info(ruta_tmp)
        duracion = info.frames / info.samplerate

        if duracion < 2.0:
            raise HTTPException(status_code=400, detail="Audio muy corto (mínimo 2 segundos).")

        # ── Normalización de amplitud ─────────────────────────────────────
        # Unity graba a nivel bajo (~0.001 peak). openSMILE no detecta voz
        # si el pico es < 0.1. Normalizamos a 0.7 peak antes de analizar.
        data, sr = sf.read(ruta_tmp, dtype="float32")
        peak = float(np.max(np.abs(data)))
        if 0 < peak < 0.15:
            data = data / peak * 0.70
            ruta_analisis = ruta_tmp + "_norm.wav"
            sf.write(ruta_analisis, data, sr)

        df = smile.process_file(ruta_analisis)
        if df.empty:
            raise HTTPException(status_code=422, detail="No se pudieron extraer features del audio.")

        fd     = {col: float(df[col].iloc[0]) for col in df.columns}
        scores = calcular_scores(fd)

        return JSONResponse({
            "scores":         scores,
            "interpretacion": interpretar(scores),
            "duracion_s":     round(duracion, 2),
            "peak_original":  round(peak, 5),
            "features_raw": {
                # pitch
                "pitch_medio":        round(fd.get("F0semitoneFrom27.5Hz_sma3nz_amean",      0), 4),
                "pitch_variabilidad": round(fd.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", 0), 4),
                # voice quality
                "jitter":             round(fd.get("jitterLocal_sma3nz_amean",               0), 6),
                "shimmer_dB":         round(fd.get("shimmerLocaldB_sma3nz_amean",            0), 4),
                "HNR_dB":             round(fd.get("HNRdBACF_sma3nz_amean",                  0), 4),
                # loudness
                "loudness_media":     round(fd.get("loudness_sma3_amean",                    0), 6),
                "loudness_variacion": round(fd.get("loudness_sma3_stddevNorm",               0), 4),
                # spectral
                "spectral_flux":      round(fd.get("spectralFlux_sma3_amean",                0), 4),
                "spectral_flux_std":  round(fd.get("spectralFlux_sma3_stddevNorm",           0), 4),
                # speech rate / pauses
                "voiced_per_sec":     round(fd.get("VoicedSegmentsPerSec",                   0), 4),
                "mean_voiced_s":      round(fd.get("MeanVoicedSegmentLengthSec",             0), 4),
                "mean_unvoiced_s":    round(fd.get("MeanUnvoicedSegmentLength",              0), 4),
                "std_unvoiced_s":     round(fd.get("StddevUnvoicedSegmentLength",            0), 4),
            }
        })

    finally:
        os.unlink(ruta_tmp)
        if ruta_analisis != ruta_tmp and os.path.exists(ruta_analisis):
            os.unlink(ruta_analisis)
