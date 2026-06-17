"""
api.py — API REST para análisis de audio y texto con IA.

ENDPOINTS:
    GET  /health         → {"status": "ok"}
    POST /analizar       → multipart WAV → 7 scores prosódicos (openSMILE)
    POST /analizar-texto → JSON transcript → scores de habilidades y pitch (Groq/Llama)
"""

import os
import json
import tempfile
import numpy as np
import soundfile as sf
import opensmile

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from groq import Groq

app = FastAPI(title="MentorIA Audio Analyzer", version="2.0")

# Inicializar openSMILE una sola vez al arrancar (costoso, no por request)
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# ─────────────────────────────────────────────────────────────────────────────
# Rangos calibrados con grabaciones reales (micrófono Unity, 16 kHz, normalizado)
# Valores observados: jitter 0.03-0.07, shimmer 0.87-1.05, HNR 6.6-7.0 dB,
#   loudness_std 0.81-0.89, voiced_per_sec ~1.7, mean_unvoiced 0.34-0.39


def n01(val, lo, hi):
    """Normaliza val a [0,1] en el rango [lo, hi], clipeado."""
    if hi == lo:
        return 0.5
    return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))


def calcular_scores(fd):
    # ── Nerviosismo (0-10) ─────────────────────────────────────────────────
    # Features confiables para micrófono: HNR (invertido) + loudness_std
    # jitter/shimmer NO se usan: amplificación de ruido los infla artificialmente
    hnr      = fd.get("HNRdBACF_sma3nz_amean",   7.0)
    loud_std = fd.get("loudness_sma3_stddevNorm", 0.60)
    hnr_inv  = 1.0 - n01(hnr,      4.0, 12.0)  # HNR bajo → más nervioso
    loud_n   = n01(loud_std, 0.40,  1.00)       # variación alta → más nervioso
    nerviosismo = round(min((hnr_inv * 0.55 + loud_n * 0.45) * 10, 10.0), 2)

    # ── Confianza (0-10) ───────────────────────────────────────────────────
    # HNR (voz limpia = confiada) + voiced rate (habla fluida = confiada)
    vps     = fd.get("VoicedSegmentsPerSec", 1.5)
    hnr_dir = n01(hnr, 4.0, 12.0)   # HNR alto → más confiado
    vps_n   = n01(vps, 0.0,  3.5)
    confianza = round(min((hnr_dir * 0.55 + vps_n * 0.45) * 10, 10.0), 2)

    # ── Energía / Entusiasmo (0-10) ────────────────────────────────────────
    loud_med = fd.get("loudness_sma3_amean",     0.30)
    flux_med = fd.get("spectralFlux_sma3_amean", 0.28)
    ln       = n01(loud_med, 0.05, 0.80)
    fn       = n01(flux_med, 0.05, 0.55)
    energia  = round(min((ln * 0.60 + fn * 0.40) * 10, 10.0), 2)

    # ── Monotonía (0-10, 10 = muy monótono) ───────────────────────────────
    # F0 stddev bajo → pitch plano → monótono
    f0_std    = fd.get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm", 0.25)
    mono_n    = float(np.clip(1.0 - (f0_std / 0.50), 0.0, 1.0))
    monotonia = round(mono_n * 10, 2)

    # ── Dinamismo vocal (0-10) ─────────────────────────────────────────────
    flux_val  = fd.get("spectralFlux_sma3_stddevNorm",
                       fd.get("spectralFlux_sma3_amean", 0.28))
    d_loud    = n01(loud_std, 0.30, 1.00)
    d_flux    = n01(flux_val, 0.05, 0.55)
    dinamismo = round((d_loud * 0.60 + d_flux * 0.40) * 10, 2)

    # ── Velocidad del habla (0-10, ~5 = normal) ───────────────────────────
    # Max real observado con micrófono Unity normalizado: ~3.5 segs vocalizados/s
    vps_val  = fd.get("VoicedSegmentsPerSec", None)
    velocidad = 5.0 if vps_val is None else round(float(np.clip(vps_val / 3.5, 0.0, 1.0)) * 10, 2)

    # ── Ratio de pausas (0-10, 10 = muchas pausas) ────────────────────────
    mun = fd.get("MeanUnvoicedSegmentLength", None)
    ratio_pausas = 5.0 if mun is None else round(float(np.clip((mun - 0.05) / 0.75, 0.0, 1.0)) * 10, 2)

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

    # Con micrófono Unity la línea base de nerviosismo es ~6-7 (HNR bajo + ruido)
    partes.append("tranquilo y controlado" if n <= 5.0 else
                  "tensión vocal moderada" if n <= 7.5 else
                  "tensión vocal alta")

    partes.append("voz confiada" if c >= 6.0 else
                  "confianza moderada"  if c >= 3.5 else
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


# ─────────────────────────────────────────────────────────────────────────────
# /analizar-texto  — análisis de transcript con Groq / Llama 3.3
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_SISTEMA = """Eres un evaluador experto en pitch de ventas y habilidades directivas.
Analiza transcripts de conversaciones y devuelve SOLO un JSON válido. Sin texto extra antes ni después."""

PROMPT_PLANTILLA = """Analiza este transcript de pitch de ventas:

CONTEXTO:
- Producto: {producto_nombre}
- Descripción: {producto_descripcion}
- Audiencia objetivo: {audiencia}

TRANSCRIPT:
{transcript}

Devuelve EXACTAMENTE este JSON (reemplazá los valores, sin texto fuera del JSON):
{{
  "habilidades_directivas": {{
    "toma_decisiones": 0,
    "pensamiento_estrategico": 0,
    "negociacion": 0,
    "liderazgo": 0,
    "orientacion_comercial": 0,
    "analisis_riesgo": 0,
    "resolucion_problemas": 0,
    "priorizacion": 0
  }},
  "elementos_pitch": {{
    "cliente_objetivo": false,
    "problema_necesidad": false,
    "solucion_propuesta": false,
    "valor_diferenciador": false,
    "justificacion_superioridad": false,
    "beneficios_concretos": false,
    "evidencia_ejemplos": false,
    "llamado_accion": false,
    "capacidad_sintesis": 0,
    "estructura_mensaje": 0
  }},
  "score_global_texto": 0,
  "feedback_principal": "máximo 2 oraciones con el feedback más útil"
}}

CRITERIOS DE EVALUACIÓN (scores 0-10, sé estricto):
- toma_decisiones: ¿propone opciones claras, elige caminos, recomienda con criterio?
- pensamiento_estrategico: ¿menciona visión a largo plazo, mercado, posicionamiento, competencia?
- negociacion: ¿adapta la propuesta, es flexible, propone condiciones o acuerdos?
- liderazgo: ¿transmite autoridad, compromiso, responsabilidad en su equipo?
- orientacion_comercial: ¿habla de ROI, costos, rentabilidad, impacto de negocio?
- analisis_riesgo: ¿menciona riesgos, garantías, contingencias, seguridad?
- resolucion_problemas: ¿identifica el problema con precisión, propone solución ejecutable?
- priorizacion: ¿menciona qué es lo más importante, qué hacer primero, foco?
- cliente_objetivo: ¿identifica claramente a quién va dirigida la solución?
- problema_necesidad: ¿describe el dolor o necesidad del cliente?
- solucion_propuesta: ¿explica cómo funciona la solución?
- valor_diferenciador: ¿menciona por qué es única o diferente a otras opciones?
- justificacion_superioridad: ¿justifica por qué es mejor que la competencia?
- beneficios_concretos: ¿menciona beneficios con números, % o ejemplos específicos?
- evidencia_ejemplos: ¿incluye casos de uso, datos, testimonios o ejemplos concretos?
- llamado_accion: ¿propone un siguiente paso o intenta cerrar la venta?
- capacidad_sintesis: 10=muy conciso y claro, 0=verborrágico y confuso
- estructura_mensaje: 10=flujo lógico problema→solución→beneficio→cierre, 0=sin estructura
- score_global_texto: promedio ponderado de todas las dimensiones"""


@app.post("/analizar-texto")
async def analizar_texto(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Body debe ser JSON válido.")

    transcript          = body.get("transcript", "").strip()
    producto_nombre     = body.get("producto_nombre", "No especificado")
    producto_descripcion = body.get("producto_descripcion", "No especificada")
    audiencia           = body.get("audiencia", "No especificada")

    if len(transcript) < 30:
        raise HTTPException(status_code=400, detail="Transcript demasiado corto (mínimo 30 caracteres).")

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY no configurada en el servidor.")

    prompt = PROMPT_PLANTILLA.format(
        producto_nombre=producto_nombre,
        producto_descripcion=producto_descripcion,
        audiencia=audiencia,
        transcript=transcript[:6000],   # límite de seguridad (~4500 tokens)
    )

    try:
        cliente = Groq(api_key=groq_key)
        respuesta = cliente.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        resultado = json.loads(respuesta.choices[0].message.content)
        return JSONResponse(resultado)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"La IA devolvió JSON inválido: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error Groq: {str(e)}")
