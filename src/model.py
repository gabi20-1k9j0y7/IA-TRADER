# src/model.py
# Inferencia + Narrativa GPT-4o

import os
import sys
import json
import math
import re
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Rutas Base
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "master"

# Configuración del Ticker y Modelo LLM
TICKER_NAME = "Amazon (AMZN)"
O4_MODEL = os.getenv("O4_MODEL", "gpt-4o")

# Mapeo de Archivos de Modelos
FILEMAP = {
    "1d": MODELS_DIR / "model_1d.joblib",
    "1m": MODELS_DIR / "model_21d.joblib",
    "1y": MODELS_DIR / "model_252d.joblib",
    "2y": MODELS_DIR / "model_504d.joblib",
}

HORIZONTE_TEXTO = {
    "1d": "la sesión de mañana",
    "1m": "el próximo mes",
    "1y": "el próximo año (12 meses)",
    "2y": "los próximos dos años"
}

# Importar SHAP
try:
    import shap
    USAR_SHAP = True
except ImportError:
    shap = None
    USAR_SHAP = False

def _to_native(obj):
    """Convierte tipos de NumPy a nativos de Python para JSON serializable."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_native(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    elif obj is None:
        return None
    else:
        return obj

def _read_maestro_with_stats(path: Path) -> Tuple[pd.Series, pd.Series, str]:
    """Lee el CSV, calcula medias históricas y devuelve la última fila como SERIES."""
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el archivo de datos: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.astype(str)

    col_date = None
    for dc in ("Date", "date", "fecha"):
        if dc in df.columns:
            try: df[dc] = pd.to_datetime(df[dc], format="%d-%m-%y", errors="coerce")
            except: df[dc] = pd.to_datetime(df[dc], dayfirst=True, errors="coerce")
            df = df.dropna(subset=[dc]).sort_values(dc).reset_index(drop=True)
            col_date = dc
            break
    
    if df.empty:
        raise ValueError("El DataFrame está vacío o no tiene fecha válida.")

    # Medias estadísticas sobre todo el histórico
    stats_mean = df.select_dtypes(include=[np.number]).mean()
    
    # Última fila para predecir
    last_row = df.iloc[-1]
    
    # Fecha formateada
    fecha_str = "Desconocida"
    if col_date:
        ts = last_row[col_date]
        if pd.notnull(ts):
            fecha_str = ts.date().isoformat()

    return last_row, stats_mean, fecha_str

def _align_features(model, row_df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    """Alinea las columnas del DF con las que espera el modelo."""
    feats = None
    if hasattr(model, "feature_names_in_"): 
        feats = list(model.feature_names_in_)
    elif hasattr(model, "get_booster"): 
        feats = list(model.get_booster().feature_names)
    else: 
        feats = row_df.select_dtypes(include=[np.number]).columns.tolist()

    feats = [str(f) for f in feats]
    X_row = row_df.copy()
    
    # Rellenar columnas faltantes con 0
    for c in feats:
        if c not in X_row.columns: 
            X_row[c] = 0.0
            
    X_row = X_row[feats]
    X_row = X_row.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return feats, X_row

def _limpiar_nombre_tecnico_natural(fname: str) -> str:
    """Traduce nombres técnicos a lenguaje natural sencillo (Validado)."""
    fname = str(fname).lower()
    
    # --- VOLATILIDAD HISTÓRICA ---
    if "hv_yz_30" in fname: return "la volatilidad del último mes"
    if "hv_yz_270" in fname: return "la volatilidad de los últimos 6 meses"
   
    
    # --- RATIOS DE VOLATILIDAD ---
    if "vol_ratio_21_252" in fname: return "la volatilidad de corto plazo vs medio plazo"
    if "vol_ratio_252_1260" in fname: return "la volatilidad de medio plazo vs largo plazo"
    if "vol_ratio" in fname: return "el ratio de volatilidad relativa"

    # --- SMA / DISTANCIA ---
    if "dist_sma" in fname:
        nums = re.findall(r'\d+', fname)
        dias = nums[-1] if nums else ""
        return f"la distancia a la media móvil de {dias} sesiones"

    if "sma_slope" in fname:
        nums = re.findall(r'\d+', fname)
        dias = nums[-1] if nums else ""
        return f"la inclinación de la media móvil de {dias} sesiones"
        
    if "sma" in fname and "dist" not in fname and "slope" not in fname:
        nums = re.findall(r'\d+', fname)
        if nums: return f"la media móvil técnica de {nums[-1]} sesiones"
        return "la media móvil técnica"

    # --- INDICADORES ESPECÍFICOS ---
    if "trend_ok" in fname: return "estado_tendencia_200"
    if "drawdown_long" in fname: return "la profundidad de las correcciones del último ciclo de 5 años"
    if "trend_slope" in fname: return "la aceleración del precio"
    if "trend" in fname: return "la tendencia general"
    
    if "from_ath" in fname: return "la distancia al máximo histórico"
    if "from_52w_high" in fname: return "la distancia al máximo de las últimas 52 semanas"
    if "from_52w_low" in fname: return "la distancia al mínimo de las últimas 52 semanas"
    
    if "rsi" in fname: return "el indicador RSI"
    if "macd" in fname: return "el impulso MACD"
    if "roc" in fname or "momentum" in fname: return "el momentum"
    if "atr" in fname: return "la volatilidad diaria"
    if "bollinger" in fname or "bb_" in fname: return "las bandas de bollinger"
    if "volume" in fname: return "el volumen"
    if "close" in fname: return "el nivel de precio"
    if "iv_" in fname: return "la volatilidad implícita"

    # --- NOTICIAS ---
    if "news" in fname:
        tipo = "la volatilidad" if "std" in fname else "el sentimiento"
        suffix = "del día" 
        if "w3" in fname: suffix = "de los últimos 3 días"
        elif "w21" in fname: suffix = "del último mes"
        elif "w63" in fname: suffix = "de los últimos 3 meses"
        elif "w126" in fname: suffix = "de los últimos 6 meses"
        return f"{tipo} de las noticias {suffix}"
    
    return fname.replace("_", " ")


def _get_market_context(row_series, stats_mean):
    """Extrae contexto de mercado (Precio, Volumen, Sentimiento, Opciones)."""
    ctx = {}
    
    # Función auxiliar para buscar en la Series
    def find_val(keyword, anti_keyword=None):
        for col in row_series.index:
            c_str = str(col).lower()
            if keyword in c_str:
                if anti_keyword and anti_keyword in c_str: continue
                val = row_series[col]
                return float(val), str(col)
        return None, None

    # Precio
    price_val, _ = find_val("close")
    if price_val is not None:
        ctx["precio"] = price_val

    # Volumen
    vol_val, vol_col = find_val("volume", "ratio")
    if vol_val is not None:
        avg = float(stats_mean.get(vol_col, vol_val))
        
        vol_m = vol_val / 1_000_000
        avg_m = avg / 1_000_000
        ctx["volumen_dato"] = f"{vol_m:.2f} Millones (Media Histórica: {avg_m:.2f} Millones)"
        
        diff_pct = ((vol_val - avg) / avg) * 100 if avg != 0 else 0
        ctx["volumen_estado"] = "superior" if diff_pct > 0 else "inferior"

    # Sentimiento de las noticias
    news_val, news_col = find_val("news_sent")
    if news_val is None: news_val, news_col = find_val("news_mean")
    
    if news_val is not None:
        if news_val <= -5: ctx["sentimiento"] = "muy pesimista"
        elif news_val < -1: ctx["sentimiento"] = "pesimista"
        elif news_val <= 1: ctx["sentimiento"] = "neutro"
        elif news_val < 5: ctx["sentimiento"] = "optimista"
        else: ctx["sentimiento"] = "muy optimista"

    # Mercado de opciones
    iv_val, iv_col = find_val("iv_atm_30")
    if iv_val is not None:
        avg = float(stats_mean.get(iv_col, iv_val))
        ctx["opciones"] = "miedo" if iv_val > avg else "calma"
    
    return ctx

def _get_shap_drivers(model, X_input, feats_list, stats_mean, pred_pct, horizon_key):
    """Calcula SHAP values y filtra lógica causal alineada con la predicción."""
    if not shap or not USAR_SHAP: return []

    try:
        explainer = None
        try: explainer = shap.TreeExplainer(model)
        except: 
            try: explainer = shap.Explainer(model)
            except: pass
        
        if explainer is None: return [] 
        
        sv = explainer(X_input)
        vals = sv.values[0] if len(sv.values.shape) == 1 else sv.values[0]
        
        all_drivers = []
        for i, val in enumerate(vals):
            fname = feats_list[i]
            impact = float(val)
            raw_val = float(X_input.iloc[0, i])
            mean_val = float(stats_mean.get(fname, 0))
            
            clean_name = _limpiar_nombre_tecnico_natural(fname)
            
            # --- LÓGICA DE ESTADO ---
            estado_relativo = "neutral"

            if "news" in fname.lower():
                if -1 <= raw_val <= 1: estado_relativo = "neutral"
                else: estado_relativo = "alto" if raw_val > mean_val else "bajo"
            
            elif "drawdown" in fname.lower():
                if raw_val < mean_val: estado_relativo = "alto"
                else: estado_relativo = "bajo"
            
            elif "trend_ok" in fname.lower():
                if raw_val > 0.5: 
                    clean_name = "estar por encima de la media móvil de 200 sesiones"
                    estado_relativo = "BINARIO_POSITIVO"
                else: 
                    clean_name = "estar por debajo de la media móvil de 200 sesiones"
                    estado_relativo = "BINARIO_NEGATIVO" 

            elif "close" in fname.lower() and "dist" not in fname.lower():
                estado_relativo = "actual"

            elif "dist" in fname.lower() or "from" in fname.lower() or "width" in fname.lower():
                if "from_ath" in fname.lower() or "high" in fname.lower():
                    # Distancias negativas a maximos
                    if raw_val > mean_val: estado_relativo = "reducida" # Cerca del max
                    else: estado_relativo = "elevada" # Lejos
                else:
                    if raw_val > mean_val: estado_relativo = "elevada"
                    else: estado_relativo = "reducida"
            
            else:
                estado_relativo = "alto" if raw_val > mean_val else "bajo"

            all_drivers.append({
                "variable": clean_name,
                "estado": estado_relativo,
                "senal": "alcista" if impact > 0 else "bajista",
                "impacto_raw": abs(impact)
            })

        all_drivers.sort(key=lambda x: x['impacto_raw'], reverse=True)
        
        # Filtrado por coherencia direccional
        target_signal = "alcista" if pred_pct > 0 else "bajista"
        drivers_favor = [d for d in all_drivers if d['senal'] == target_signal]
        
        selected = []
        threshold = 0.5 if horizon_key in ["1d", "1m"] else 5.0 

        if abs(pred_pct) > threshold:
            selected = drivers_favor[:2]
        else:
            # Si hay poca direccionalidad muestra fuerzas opuestas
            top_bull = next((d for d in all_drivers if d['senal'] == 'alcista'), None)
            top_bear = next((d for d in all_drivers if d['senal'] == 'bajista'), None)
            if top_bull: selected.append(top_bull)
            if top_bear: selected.append(top_bear)

        return [{"variable": d["variable"], "estado": d["estado"], "senal": d["senal"]} for d in selected]

    except Exception:
        return []

# PROMPTS

def _generate_prompts(hkey: str, payload: dict) -> Tuple[str, str]:
    """Genera los prompts validados para el modelo LLM."""
    
    # Preparar JSON seguro
    payload_safe = _to_native(payload)
    json_str = json.dumps(payload_safe, ensure_ascii=False, indent=2)
    user_prompt = f"Datos:\n```json\n{json_str}\n```"

    formatting_rules = (
        "REGLAS DE ESTILO:\n"
        "- Extensión TOTAL: 80-100 palabras.\n"
        "- Tono: Natural y fluido. NADA de comillas, mayúsculas raras ni jerga técnica.\n"
        "- PROHIBIDO: Usar palabras como 'nuestro', 'mi', 'nosotros'. Usa SIEMPRE tercera persona ('el algoritmo', 'el modelo').\n"
        "- Estructura: 3 párrafos separados por línea en blanco.\n"
    )

    p1_inst = (
        "P1 (Hechos): 'En la última sesión...'. \n"
        "- Indica el precio de cierre.\n"
        "- Indica el volumen EXACTO y su media histórica (usa el dato 'volumen_dato' del JSON). NO interpretes el volumen.\n"
        "- Menciona el sentimiento de las 'Noticias del día' y la situación del 'mercado de opciones' (usa este término exacto).\n"
        "- IMPORTANTE: Sé meramente descriptivo. NO hagas juicios de valor ('calma', 'estabilidad').\n"
    )

    p2_base = (
        "P2 (Predicción y Porqués): 'El algoritmo predice...'. Cita el %.\n"
        "Explica los factores clave usando OBLIGATORIAMENTE esta lógica causal:\n"
        "- Si el factor es 'estar por encima/debajo de la media...', ÚSALO TAL CUAL. NO digas 'el nivel de estar por...'. Di: 'El modelo interpreta que estar por encima... actúa como señal... para este horizonte temporal'.\n"
        
        # --- REGLA ESTRICTA DE ESTRUCTURA Y VERBOS ---
        "- ESTRUCTURA OBLIGATORIA: '[Nombre Variable], que [Verbo] [Estado], actúa como...'.\n"
        "- PROHIBIDO convertir el estado en adjetivo directo (Ej: NO digas 'la pendiente alta', di 'la pendiente, que es alta').\n"
        
        "- EXCEPCIÓN 'actual': Si el estado proporcionado en el JSON es 'actual', NO digas ', que es actual'. OMITELO. Di simplemente: 'El [Variable] actúa como...'.\n"

        "- ELECCIÓN DE VERBO (Para el resto):\n"
        "  1. Si la variable implica un periodo pasado o histórico (ej: 'último mes', 'media histórica', 'trayectoria', 'volatilidad histórica'), usa: 'ha sido'.\n"
        "  2. Si la variable es un dato técnico puntual o estructural actual (ej: 'nivel', 'pendiente', 'distancia', 'RSI', 'Momentum'), usa: 'es' o 'se encuentra'.\n"
        # ---------------------------------------------

        "- Para el resto: 'El modelo interpreta que el nivel [estado] de [variable] actúa como señal [senal] para este horizonte temporal'.\n"
        "- PROHIBIDO inventar factores que no estén en la lista 'factores_clave'. Si solo hay uno, explica solo uno.\n"
        "- PROHIBIDO usar noticias o contexto para explicar la predicción si no están en 'factores_clave'.\n"
    )

    if hkey == "1d":
        structure = (
            f"{p1_inst}\n"
            "P2 (Educativo y Datos):\n"
            "- Empieza explicando que los movimientos a 1 día son 'ruido de mercado' (Random Walk), caóticos e imposibles de predecir.\n"
            "- Aclara que 'el algoritmo' no realiza una predicción direccional a un día, sino que proyecta la rentabilidad media histórica como una referencia estadística neutral.\n"
            "- Menciona que la media (sin predicha ni proyectada, a secas) es 'prediccion_pct' (Copia el valor EXACTO del JSON, incluido el %, no redondees).\n\n"
            "P3 (Conclusión):\n"
            "- Advertencia de prudencia: en el corto plazo domina el azar."
        )
        role = "Analista Cuantitativo"

    elif hkey == "1m":
        structure = (
            f"{p1_inst}\n"
            f"{p2_base}"
            "- IMPORTANTE: Enfoca la explicación en la inversión a corto plazo.\n\n"
            "P3 (Conclusión): Veredicto operativo."
        )
        role = "Gestor de Fondos"

    else:
        structure = (
            f"{p1_inst}\n"
            f"{p2_base}"
            "- IMPORTANTE: Enfoca la explicación en el largo plazo. NUNCA menciones el corto plazo.\n\n"
            "P3 (Conclusión): Veredicto operativo."
        )
        role = "Gestor de Fondos"

    system_prompt = f"Eres {role}. Analizas {TICKER_NAME} a {HORIZONTE_TEXTO[hkey]}.\n{formatting_rules}\n{structure}"

    return system_prompt, user_prompt


# Función principal
def run_inference(
    hkey: str,
    maestro_path: str, # Puede ser str o Path
    use_llm: bool = True
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo: Carga datos -> Modelo -> SHAP -> LLM -> Resultado.
    """
    if hkey not in FILEMAP:
        raise ValueError(f"Horizonte no válido: {hkey}")

    path_obj = Path(maestro_path)
    
    # Cargar Datos y Estadísticas
    row_series, stats_mean, fecha_str = _read_maestro_with_stats(path_obj)
    
    # Cargar Modelo
    model_path = FILEMAP[hkey]
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    model = joblib.load(model_path)
    
    # Preparar Input del Modelo
    # Como row_series es una Series, .to_frame().T funciona correctamente para crear un DataFrame de una fila (1, n_features)
    feats_list, X_input = _align_features(model, row_series.to_frame().T)

    # Inferencia Numérica
    yhat_log = float(model.predict(X_input)[0])
    pred_pct = (math.exp(yhat_log) - 1.0) * 100.0
    
    # Contexto y Drivers
    contexto = _get_market_context(row_series, stats_mean)
    
    # Calculo de drivers excepto para un horizonte de 1 día
    drivers = []
    if hkey != "1d":
        drivers = _get_shap_drivers(model, X_input, feats_list, stats_mean, pred_pct, hkey)

    # Preparar Payload LLM
    decimales = 3 if hkey == "1d" else 2
    
    payload = {
        "activo": TICKER_NAME,
        "fecha": fecha_str,
        "horizonte": HORIZONTE_TEXTO[hkey], 
        "horizonte_txt": HORIZONTE_TEXTO[hkey],
        "prediccion_pct": f"{round(pred_pct, decimales)}%", 
        "datos_mercado": contexto,
        "factores_clave": drivers
    }

    # Llamada a OpenAI
    narrative = None
    llm_diag = {"used": False}

    if use_llm:
        sys_p, user_p = _generate_prompts(hkey, payload)
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                
                resp = client.chat.completions.create(
                    model=O4_MODEL,
                    messages=[
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": user_p}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )
                narrative = resp.choices[0].message.content.strip()
                llm_diag["used"] = True
                llm_diag["model"] = O4_MODEL
            except Exception as e:
                llm_diag["error"] = str(e)
        else:
             llm_diag["error"] = "No API Key found"

 
# Retorno Estructurado
    return {
        "date": fecha_str,
        "horizon_days": hkey, 
        "pred_pct": pred_pct,        
        "ui_pct": round(pred_pct, decimales), 
        "narrative": narrative,
        "llm_used": llm_diag["used"],
        "llm_diag": llm_diag,
        "payload_debug": payload
    }
