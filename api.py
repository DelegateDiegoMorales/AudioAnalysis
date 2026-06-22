"""
api.py — API REST para análisis de audio y texto con IA.

ENDPOINTS:
    GET  /health             → {"status": "ok"}
    GET  /reporte             → HTML del reporte de evaluación (recibe datos vía hash #base64json,
                                 o en modo en vivo si se abre sin hash — ver /reporte/ultimo)
    POST /reporte/publicar    → Unity publica acá el JSON del último reporte analizado
    GET  /reporte/ultimo      → último reporte publicado (polling desde reporte.html)
    POST /analizar            → multipart WAV → 7 scores prosódicos (openSMILE)
    POST /analizar-texto      → JSON transcript → rúbrica pedagógica 4 dimensiones (Groq)
"""

import os
import json
import tempfile
import threading
import numpy as np
import soundfile as sf
import opensmile

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
import requests as req_lib

app = FastAPI(title="MentorIA Audio Analyzer", version="2.0")

# ─────────────────────────────────────────────────────────────────────────────
# Estado del "último reporte publicado" — en memoria, no persiste entre
# reinicios del proceso (si Render redespliega o el free-tier se duerme y
# despierta, se pierde). Es intencional: solo sirve para que la página
# /reporte en modo en vivo (sin hash) pueda mostrar la sesión más reciente
# sin que el dispositivo VR (Quest) tenga que abrir un navegador — Unity
# manda el JSON acá por POST, y cualquier otra pantalla con /reporte
# abierto lo va a ver solo, vía polling.
# ─────────────────────────────────────────────────────────────────────────────
_ultimo_reporte_lock = threading.Lock()
_ultimo_reporte: dict | None = None

# Contador incremental — se le pega un "_pub_id" único a cada reporte
# publicado, independiente de su contenido. reporte.html lo usa para saber
# con certeza si hay un reporte NUEVO que mostrar, en vez de adivinar a
# partir de nombre/fecha/intento/puntaje_total (que pueden repetirse entre
# pruebas distintas — ej. mismo nombre default, mismo día, mismo puntaje —
# y antes hacían que un reporte nuevo se confundiera con el anterior y no
# se refrescara la pantalla que ya estaba abierta).
_ultimo_reporte_seq = 0

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


@app.get("/reporte", response_class=HTMLResponse)
def reporte():
    """
    Sirve la página HTML de reporte de evaluación.
    Unity llama: Application.OpenURL(apiUrl + "/reporte#" + Base64(jsonData))
    El HTML lee el hash y renderiza el reporte sin necesidad de estado en servidor.
    """
    ruta = os.path.join(os.path.dirname(__file__), "reporte.html")
    if not os.path.exists(ruta):
        raise HTTPException(status_code=404, detail="reporte.html no encontrado junto a api.py")
    with open(ruta, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/reporte/publicar")
async def publicar_reporte(request: Request):
    """
    Unity manda acá el JSON completo del reporte (mismo shape que arma
    ReporteBuilder.ConstruirJson en el cliente) apenas termina de analizar
    una sesión — no hace falta abrir ningún navegador desde el Quest, el
    dato queda disponible para quien tenga /reporte abierto en otra
    pantalla (vía /reporte/ultimo, con polling).
    """
    global _ultimo_reporte, _ultimo_reporte_seq
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON inválido")

    with _ultimo_reporte_lock:
        _ultimo_reporte_seq += 1
        data["_pub_id"] = _ultimo_reporte_seq
        _ultimo_reporte = data

    return {"ok": True, "pub_id": _ultimo_reporte_seq}


@app.get("/reporte/ultimo")
def obtener_ultimo_reporte():
    """
    Devuelve el último reporte publicado, o {"sin_datos": true} si todavía
    no hay ninguno.

    Antes esto devolvía status_code=204 con content=None. Starlette igual
    serializa ese None a un body de 4 bytes ("null") en un JSONResponse,
    pero uvicorn no permite NINGÚN body en una respuesta 204 (es inválido
    por spec HTTP) — eso producía "RuntimeError: Response content longer
    than Content-Length" en cada poll sin reporte todavía, tirando abajo
    el request ASGI a la mitad y pudiendo romper la conexión/el polling
    de la página abierta. Usar siempre 200 con un campo "sin_datos" evita
    el problema de raíz en vez de pelear con las reglas de 204 sin body.
    """
    with _ultimo_reporte_lock:
        if _ultimo_reporte is None:
            return JSONResponse(content={"sin_datos": True})
        return JSONResponse(content=_ultimo_reporte)


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
# /analizar-texto  — rúbrica pedagógica 4 dimensiones (Groq)
#
# Basada en la rúbrica de Catalina / USS:
#   4 dimensiones × máx 5 pts = 20 pts total
#   Básico=1pt, Intermedio=3pts, Avanzado=5pts
#   + KPIs de pitch (booleanos con evidencia)
#   + Competencias empresariales (3 niveles)
#   + Feedback en 3 bloques pedagógicos
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_SISTEMA = """Eres un evaluador pedagógico experto en pitch de emprendimiento.
Analizas transcripts y devuelves SOLO un JSON válido, sin texto extra antes ni después."""

PROMPT_PLANTILLA = """Evalúa este transcript de pitch de emprendimiento según la rúbrica pedagógica indicada.

CONTEXTO DEL EJERCICIO:
- Producto / Emprendimiento: {producto_nombre}
- Descripción: {producto_descripcion}
- Audiencia: {audiencia}

TRANSCRIPT:
{transcript}

═══════════════════════════════════════════════════════════
RÚBRICA DE EVALUACIÓN — devuelve puntajes 1, 3 o 5 únicamente
═══════════════════════════════════════════════════════════

DIMENSIÓN 1 — INICIO DEL PITCH (máx 5 pts)
Evalúa: nombre del emprendimiento/producto, cliente objetivo, problema, solución.
  Básico    (1): No nombra producto, no define cliente, problema ni solución.
  Intermedio(3): Nombra algo, pero no conecta claramente los 4 elementos.
  Avanzado  (5): Presenta nombre, cliente objetivo, problema y solución con claridad.

DIMENSIÓN 2 — DESARROLLO Y PROPUESTA DE VALOR (máx 5 pts)
Evalúa: por qué la solución es mejor que alternativas, beneficios concretos con ejemplos.
  Básico    (1): Nombra el problema o características, sin justificar valor ni dar beneficios.
  Intermedio(3): Justifica parcialmente, pero beneficios generales o sin ejemplos concretos.
  Avanzado  (5): Justifica superioridad, nombra beneficios concretos y usa ejemplos o datos.

DIMENSIÓN 3 — CIERRE, OBJECIONES Y ORIENTACIÓN COMERCIAL (máx 5 pts)
Evalúa: llamado a la acción, respuesta a preguntas/objeciones, orientación comercial/negociación.
  Básico    (1): No cierra, no responde objeciones, evade o no hace pedido al inversionista.
  Intermedio(3): Responde algunas objeciones, pero sin cierre concreto o negociación débil.
  Avanzado  (5): Cierra con solicitud concreta, responde objeciones y negocia con criterio.

DIMENSIÓN 4 — COMUNICACIÓN, PRESENCIA Y MANEJO DE PRESIÓN (máx 5 pts)
Evalúa: claridad verbal, ritmo, muletillas, seguridad, escucha activa, control emocional.
  Básico    (1): Discurso confuso, muchas muletillas, ritmo inadecuado, alto nerviosismo.
  Intermedio(3): Comunicación comprensible, pausas, muletillas moderadas, seguridad variable.
  Avanzado  (5): Habla claro, buen ritmo, seguridad, escucha activa, manejo de presión.

═══════════════════════════════════════════════════════════
KPIs DE PITCH — detectar presencia (logrado/en_desarrollo/por_reforzar)
═══════════════════════════════════════════════════════════
  logrado:       el elemento aparece claramente en el transcript
  en_desarrollo: aparece parcialmente o con poca claridad
  por_reforzar:  ausente, muy débil o confuso

KPI_01 cliente_objetivo      — ¿nombra quién compraría o usaría el producto?
KPI_02 problema_necesidad    — ¿describe el dolor o necesidad que resuelve?
KPI_03 solucion_propuesta    — ¿explica cómo funciona la solución?
KPI_04 propuesta_valor       — ¿menciona qué lo hace valioso o único?
KPI_05 por_que_es_mejor      — ¿justifica superioridad sobre alternativas actuales?
KPI_06 beneficios_concretos  — ¿nombra beneficios con datos, números o ejemplos?
KPI_07 evidencia_ejemplos    — ¿incluye casos, testimonios, pilotos o datos reales?
KPI_08 llamado_accion        — ¿hace un pedido concreto al inversionista?

═══════════════════════════════════════════════════════════
COMPETENCIAS EMPRESARIALES — evaluar con logrado/en_desarrollo/por_reforzar
═══════════════════════════════════════════════════════════
  orientacion_comercial   — habla de ROI, costos, rentabilidad o impacto económico
  negociacion             — propone condiciones, es flexible, adapta la oferta
  pensamiento_estrategico — visión a largo plazo, mercado, posicionamiento
  toma_decisiones         — propone caminos claros, recomienda con criterio
  priorizacion            — identifica lo más importante, propone foco
  analisis_riesgo         — menciona riesgos, contingencias o garantías
  liderazgo               — transmite autoridad, compromiso, responsabilidad
  etica_profesional       — transparente, honesto, responsable en su discurso

═══════════════════════════════════════════════════════════
DEVUELVE EXACTAMENTE ESTE JSON (sin texto fuera del JSON):
═══════════════════════════════════════════════════════════
{{
  "dimensiones": {{
    "inicio_pitch":         0,
    "desarrollo_propuesta": 0,
    "cierre_objeciones":    0,
    "comunicacion_presencia": 0
  }},
  "pitch_items": {{
    "cliente_objetivo":     "por_reforzar",
    "problema_necesidad":   "por_reforzar",
    "solucion_propuesta":   "por_reforzar",
    "propuesta_valor":      "por_reforzar",
    "por_que_es_mejor":     "por_reforzar",
    "beneficios_concretos": "por_reforzar",
    "evidencia_ejemplos":   "por_reforzar",
    "llamado_accion":       "por_reforzar"
  }},
  "competencias_items": {{
    "orientacion_comercial":  "por_reforzar",
    "negociacion":            "por_reforzar",
    "pensamiento_estrategico":"por_reforzar",
    "toma_decisiones":        "por_reforzar",
    "priorizacion":           "por_reforzar",
    "analisis_riesgo":        "por_reforzar",
    "liderazgo":              "por_reforzar",
    "etica_profesional":      "por_reforzar"
  }},
  "puntaje_total": 0,
  "porcentaje": 0,
  "nivel_global": "Básico",
  "feedback": {{
    "fortaleza":          "1 oración: lo que hizo bien",
    "mejora_prioritaria": "1 oración: lo más urgente a mejorar",
    "proximo_intento":    "1 oración: recomendación concreta para la siguiente práctica"
  }}
}}

REGLAS:
- puntaje_total = suma de los 4 valores de dimensiones (min 4, max 20)
- porcentaje = round(puntaje_total / 20 * 100)
- nivel_global: "Básico" si puntaje_total ≤ 9 | "Intermedio" si 10–17 | "Avanzado" si ≥ 18
- dimensiones solo puede tener valores 1, 3 o 5
- pitch_items y competencias_items solo pueden ser "logrado", "en_desarrollo" o "por_reforzar"
- feedback debe ser breve, pedagógico y accionable (no punitivo)
- feedback NUNCA puede quedar vacío ni faltar — completá fortaleza, mejora_prioritaria y
  proximo_intento siempre, aunque el pitch sea muy básico. Es el campo MÁS importante de
  toda la respuesta: si te queda poco espacio, recortá detalle en otros campos, pero
  nunca dejes feedback incompleto.
- Si el transcript incluye Q&A, evaluar la Dimensión 3 (cierre/objeciones) también con esas respuestas"""


@app.post("/analizar-texto")
async def analizar_texto(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Body debe ser JSON válido.")

    transcript           = body.get("transcript", "").strip()
    producto_nombre      = body.get("producto_nombre",      "No especificado")
    producto_descripcion = body.get("producto_descripcion", "No especificada")
    audiencia            = body.get("audiencia",            "No especificada")

    if len(transcript) < 30:
        raise HTTPException(status_code=400, detail="Transcript demasiado corto (mínimo 30 caracteres).")

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY no configurada en el servidor.")

    prompt = PROMPT_PLANTILLA.format(
        producto_nombre=producto_nombre,
        producto_descripcion=producto_descripcion,
        audiencia=audiencia,
        transcript=transcript[:6000],
    )

    try:
        respuesta = req_lib.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model": "openai/gpt-oss-120b",
                "messages": [
                    {"role": "system", "content": PROMPT_SISTEMA},
                    {"role": "user",   "content": prompt},
                ],
                "temperature":     0.1,
                "max_tokens":      3000,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        respuesta.raise_for_status()
        resultado = json.loads(respuesta.json()["choices"][0]["message"]["content"])
        return JSONResponse(resultado)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"La IA devolvió JSON inválido: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error Groq: {str(e)}")
