# app.py
import openai
import os, time, sys, json, subprocess
import pandas as pd
import streamlit as st
from pandas import Timestamp
from pandas.tseries.offsets import BDay
from src import data, sentiment, model as infer
import importlib

infer = importlib.reload(infer)

fmt = lambda d: pd.to_datetime(d).strftime("%d-%m-%y") if pd.notna(pd.to_datetime(d, errors="coerce")) else "—"

def _limpiar_archivos_temporales(folder: str):
    import glob
    # Patrones de archivos "basura" que queremos borrar
    patrones = [
        "guardian_*.csv",
        "nyt_*.csv",
        "eodhd_*.csv",        
    ]
    
    count = 0
    for pat in patrones:
        # Busca archivos que coincidan con el patrón en la carpeta
        files = glob.glob(os.path.join(folder, pat))
        for f in files:
            try:
                os.remove(f)
                count += 1
            except Exception:
                pass
    return count

def _parse_date_col(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, format="%d-%m-%y", errors="coerce")
    if dt.notna().sum() == 0:
        dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return dt

def _generar_explicacion_simple(texto_tecnico: str):
    """Genera una versión simplificada del análisis usando OpenAI (GPT-4o)."""
    
    prompt_system = """
    Eres un amigo de confianza traduciendo un análisis financiero complejo a lenguaje de calle.
    
    TU MISIÓN:
    Lee el texto técnico y dale un consejo directo al usuario. NO uses plantillas fijas, varía tu forma de hablar, pero mantén siempre el mismo mensaje de fondo.

    REGLAS DE ORO:
    1. CERO NÚMEROS: Prohibido mencionar precios, porcentajes o fechas exactas.
    2. CERO TECNICISMOS: Nada de "momentum", "señal", "volatilidad".
    3. LENGUAJE: Natural, cercano pero bien escrito
    4. ANTI-REPETICIÓN: ESTÁ PROHIBIDO EMPEZAR SIEMPRE CON LA PALABRA "MIRA". 
       - Varía tus inicios: "La cosa está así...", "Si fuera mi dinero...", "Te cuento...", "Ojo con esto...".
       - Sorpréndeme con la variedad.
       - Redactalo bien en Español de España

    GUÍA DE CONSEJOS (TRANSMITE ESTAS IDEAS CON TUS PROPIAS PALABRAS):

    - SI ES A 1 DÍA:
      La idea clave es que es puro azar. Compáralo con ir al casino, jugar a la ruleta o echar una quiniela. Diles que eso no es invertir, es apostar.

    - SI ES A 1 MES:
      (Si es positivo): Diles que parece buena oportunidad para rascar algo de dinero rápido, pero que anden con ojo y no se despisten.
      (Si es negativo): Diles claramente que se olviden, que guarden la cartera y se ahorren el disgusto porque pinta feo a corto plazo.

    - SI ES A 1 AÑO:
      (Si es positivo): Diles que pinta a que será un buen año. Si pueden dejar el dinero quieto los próximos 12 meses, parece que crecerá.
      (Si es negativo): Diles que un año pasa volando y, si van a necesitar esa pasta pronto, mejor no arriesgarse porque pinta a bajada.

    - SI ES A 2 AÑOS O MÁS:
      (Si es positivo): Transmite que es el sitio ideal para "aparcar" el dinero, olvidarse una buena temporada y dejar que madure solo a largo plazo.
      (Si es negativo): Diles que si aprecian sus ahorros, busquen otro sitio más tranquilo. Que ahí hay riesgo estructural y mejor no jugársela.

    IMPORTANTE: Sé fiel a si el texto dice que sube o baja. No inventes la predicción, solo traduce el estilo.
    """
    
    try:
        # Asegúrate de tener OPENAI_API_KEY configurada en tu entorno
        response = openai.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": f"Traduce esto: '{texto_tecnico}'"}
            ],
            temperature=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"No se pudo simplificar: {e}"

# --- helpers para scripts externos ---
def _script_path(rel: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "src", "news", rel)

def _norm_title(t: str) -> str:
    import re
    t = (t or "").lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9 ]+", "", t)
    return t.strip()

def _run_news_script(path: str, args: list, progress_cb=None):
    if not os.path.exists(path):
        return None, f"Script no encontrado: {path}"

    proc = subprocess.Popen(
        [sys.executable, path] + args,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1
    )

    last_json_line = None
    while True:
        line = proc.stderr.readline()
        if line:
            line = line.strip()
            if line.startswith("%%") and progress_cb:
                try:
                    head, msg = line[2:].split("|", 1)
                    pct = int(head)
                    progress_cb(pct, msg)
                except Exception:
                    pass
        else:
            if proc.poll() is not None:
                break

    out = proc.stdout.read() if proc.stdout else ""
    for ln in [x for x in out.strip().splitlines() if x.strip()]:
        last_json_line = ln

    rc = proc.returncode
    if rc != 0:
        return None, f"El script terminó con rc={rc}."

    try:
        meta = json.loads(last_json_line) if last_json_line else None
        return meta, None
    except Exception:
        return None, "No se pudo parsear la salida JSON del script."

def _unify_news(csv_paths: list[str], out_dir: str, dt_from: str, dt_to: str) -> str:
    cols = ["Fecha","Titular","Resumen","URL","Seccion","Desk","Tipo","Autor","Fuente"]
    frames = []
    for p in csv_paths:
        if p and os.path.exists(p):
            try:
                df = pd.read_csv(p)
                for c in cols:
                    if c not in df.columns: df[c] = ""
                frames.append(df[cols].copy())
            except Exception:
                pass
    if not frames:
        raise RuntimeError("No hay CSVs de entrada para unificar.")

    dfu = pd.concat(frames, ignore_index=True)
    dfu["__tnorm"] = dfu["Titular"].map(_norm_title)
    dfu = dfu.sort_values("Fecha").drop_duplicates(subset=["URL"], keep="first")
    dfu = dfu.sort_values("Fecha").drop_duplicates(subset=["__tnorm"], keep="first")
    dfu = dfu.drop(columns=["__tnorm"])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"NOTICIAS_UNIFICADAS_V3_raw_{dt_from}_{dt_to}.csv")
    dfu.to_csv(out_path, index=False, encoding="utf-8")
    return out_path

def _last_master_news_day(master_csv: str):
    import pandas as pd, os
    if not os.path.exists(master_csv):
        return None
    try:
        dfm = pd.read_csv(master_csv, usecols=["Date", "news_sent_mean"])
    except Exception:
        dfm = pd.read_csv(master_csv)
        if "Date" not in dfm.columns or "news_sent_mean" not in dfm.columns:
            return None
        dfm = dfm[["Date", "news_sent_mean"]]

    dt1 = pd.to_datetime(dfm["Date"], format="%d-%m-%y", errors="coerce")
    dt2 = pd.to_datetime(dfm.loc[dt1.isna(), "Date"], errors="coerce", dayfirst=True)
    dfm["__Date"] = dt1.fillna(dt2)

    sent = pd.to_numeric(dfm["news_sent_mean"], errors="coerce")
    mask = dfm["__Date"].notna() & sent.notna()
    if not mask.any():
        return None
    return dfm.loc[mask, "__Date"].max().date()

def _compute_news_range(master_csv: str, fallback_days: int = 60):
    import pandas as pd
    today = pd.Timestamp.today().normalize().date()
    last_cov = _last_master_news_day(master_csv)
    start = (pd.Timestamp(last_cov) + pd.Timedelta(days=1)).date() if last_cov else \
            (pd.Timestamp.today().normalize() - pd.Timedelta(days=fallback_days)).date()
    end = today
    if start > end:
        start = end
    return start, end


st.set_page_config(page_title="IA Trader", layout="centered")

# --- BLOQUE DE SEGURIDAD ---
def check_password():
    """Devuelve True si la contraseña es correcta."""
    
    # 1. Definir la función de verificación
    def password_entered():
        # Compara la entrada del usuario con el secreto guardado
        if st.session_state["password"] == st.secrets["PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Borrar del estado por seguridad
        else:
            st.session_state["password_correct"] = False

    # 2. Si ya está validado, retornar True inmediatamente
    if st.session_state.get("password_correct", False):
        return True

    # 3. Mostrar input de contraseña si no está validado
    st.title("🔒 Acceso Restringido")
    st.markdown("Por favor, introduce la contraseña para acceder a **IA Trader**.")
    
    st.text_input(
        "Contraseña", 
        type="password", 
        on_change=password_entered, 
        key="password"
    )
    
    # Mensaje de error si falla
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("⛔ Contraseña incorrecta")

    return False

if not check_password():
    st.stop()  # DETIENE LA APP AQUÍ si no hay login

# --- CSS MEJORADO ---
st.markdown("""<style>
/* 1. Subir todo el contenido (reducir padding top) */
.block-container {
    max-width: 1000px !important;
    padding-top: 1rem !important;  /* Antes era 2rem */
    padding-bottom: 2rem !important;
    padding-left: 5rem !important;
    padding-right: 5rem !important;
}

/* 2. Botones Primarios (Ejecutar y Reset) -> VERDES */
button[kind="primary"] {
    background-color: #14532d !important;
    color: white !important;
    border: 0 !important;
    transition: .15s;
}
button[kind="primary"]:hover {
    background-color: #166534 !important;
}

/* 3. Botón Secundario (Explicación) -> Transparente/Gris */
button[kind="secondary"] {
    background-color: transparent !important;
    color: #cbd5e1 !important;
    border: 1px solid #475569 !important;
}
button[kind="secondary"]:hover {
    border-color: #94a3b8 !important;
    color: white !important;
}

/* Color de la barra de progreso */
[data-testid="stProgress"] > div > div > div > div { background:#14532d; }
</style>
""", unsafe_allow_html=True)

EMPRESAS = {
    "Amazon": "AMZN",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Alphabet (Google)": "GOOG",
    "Meta": "META"
}

HORIZONTES = ["1 día","1 mes","1 año","2 años"]
BDAY_MAP = {"1 día":1,"1 mes":21,"1 año":252,"2 años":504}
HKMAP = {"1 día":"1d","1 mes":"1m","1 año":"1y","2 años":"2y"}

# --- INICIALIZACIÓN DEL ESTADO ---
if "running" not in st.session_state: st.session_state.running = False
if "done" not in st.session_state: st.session_state.done = False
if "result" not in st.session_state: st.session_state.result = {}
if "simple_text" not in st.session_state: st.session_state.simple_text = None

# --- TÍTULO CON BOTÓN DE INFO ---
# --- TÍTULO CON BOTÓN DE INFO ---
# Ajustamos columnas: 85% título, 15% botón para que tenga espacio
col_title, col_info = st.columns([0.85, 0.15]) 

with col_title:
    st.title("IA Trader")

with col_info:
    # Aumentamos a 45px para bajar el botón y alinearlo con el texto
    st.markdown("<div style='height: 45px;'></div>", unsafe_allow_html=True)
    
    # El botón popover con el texto largo
    with st.popover("ℹ️"):
        st.markdown("""
    ### 🤖 Bienvenido a IA Trader
    
    **Introducción**    
    IA Trader es una aplicación de apoyo a la toma de decisiones de inversión. A partir de la situación actual del mercado, genera una proyección de la evolución del precio de la acción seleccionada y la acompaña de una explicación estructurada y comprensible para el inversor.

    **Arquitectura y Modelo**    
    El núcleo del sistema ejecuta un algoritmo **XGBoost**. Esta arquitectura fue la elegida tras entrenar y descartar otras **7 familias de modelos** (incluyendo Redes Neuronales GRU, Perceptrón Multicapa, Random Forest y modelos lineales), al demostrar una mayor capacidad para generalizar patrones fuera de la muestra (*out-of-sample*).

    **Fuente de datos**    
    El modelo se entrenó a partir del análisis profundo de la década **2016-2025**, procesando una matriz de **más de 111.000 puntos de análisis**. Esta estructura se compuso de **48 variables** predictivas (incluyendo indicadores básicos y múltiples variables derivadas) que integraban tres fuentes simultáneas. Algunas de ellas son:
    1.  **Datos Técnicos:** Se calcularon métricas clásicas de precio, volumen, medias móviles, volatilidad histórica etc.
    2.  **Análisis de sentimiento de noticias con LLM:** Un modelo de lenguaje analizó y puntuó miles de titulares globales (The Guardian, NYT, EODHD) para cuantificar la polaridad (optimismo/pesimismo) del mercado.
    3.  **Mercado de Opciones:** Se construyeron variables derivadas de la actividad en opciones (volatilidad implícita, *skew* y ratio Put/Call) que resumían las expectativas de riesgo de los participantes.

    **Guía de Horizontes**
    * **⚡ 1 Día:** Enfocado en el movimiento muy corto plazo, capturando ruido y variaciones inmediatas del mercado.  
    * **📅 1 Mes:** Ventana táctica de corto plazo, donde el modelo combina distintas señales para identificar posibles desajustes temporales en el precio.  
    * **📈 1 a 2 Años:** Visión estratégica, orientada a patrones de fondo y movimientos de ciclo más largos en la acción.

    **Proceso de predicción**                    
    En cada ejecución, IA Trader realiza automáticamente los siguientes pasos:
    1. **Actualiza datos técnicos:** Descarga los datos de precio y volumen más recientes de la acción seleccionada.
    2. **Recoge noticias relevantes:** Consulta las últimas noticias financieras y calcula indicadores de sentimiento mediante un LLM.
    3. **Integra datos de opciones:** Obtiene y actualiza métricas derivadas del mercado de opciones (volatilidad implícita, *skew*, ratio Put/Call...).
    4. **Genera las variables de entrada:** A partir de toda esta información construye las 48 variables de entrada para el horizonte elegido.
    5. **Infiere y explica:** El modelo XGBoost genera la predicción y, a partir de sus impulsores, un LLM construye un texto explicativo que presenta la señal de forma clara y argumentada.

    ---
    **Nota de Versión:** En esta primera iteración, la herramienta opera exclusivamente con acciones de **Amazon (AMZN)**.

    *⚠️ **Descargo de responsabilidad:** Esta herramienta es un soporte para la toma de decisiones y no constituye asesoramiento financiero profesional. Los mercados conllevan riesgos y el rendimiento pasado no garantiza resultados futuros.*
            """)

col1,col2,col3 = st.columns([1,1,0.8])
with col1:
    st.markdown("**Empresa**")
    nombre_empresa = st.selectbox("Empresa", list(EMPRESAS.keys()), index=0, label_visibility="collapsed")
    ticker = EMPRESAS[nombre_empresa]
    empresa_valida = (nombre_empresa == "Amazon")

with col2:
    st.markdown("**Horizonte de inversión**")
    horizonte = st.selectbox("Horizonte de inversión", HORIZONTES, index=1, label_visibility="collapsed")

with col3:
    st.markdown("<div style='height:41px'></div>", unsafe_allow_html=True)
    boton_desactivado = st.session_state.running or (not empresa_valida)
    # AÑADIDO type="primary" para activar el CSS VERDE
    run = st.button("Ejecutar", type="primary", use_container_width=True, disabled=boton_desactivado)


# AÑADIDO type="primary" para que Reset también sea VERDE
reset = st.button("Reset", type="primary")
if reset:
    st.session_state.running = False
    st.session_state.done = False
    st.session_state.result = {}
    st.session_state.simple_text = None
    st.rerun()

if not empresa_valida:
    st.warning("⚠️ Empresa no disponible en esta versión")

MASTER = f"data/master/{ticker}_Maestro_Inference.csv"

if run:
    st.session_state.running = True
    st.session_state.done = False
    st.session_state.result = {}
    st.session_state.simple_text = None
    st.rerun()

def make_stage_progress(progress, msg, start_pct, end_pct):
    span = max(1, end_pct - start_pct)
    def _cb(pct, _text):
        scaled = int(start_pct + (pct * span) / 100)
        progress.progress(scaled)
        time.sleep(1.0)
    return _cb

zona_resultados = st.empty()
if st.session_state.running:
    st.caption("⚠️ Puede durar unos minutos.")
    progress = st.progress(0, text="Progreso")
    msg = st.empty()

    etapas = [
        f"Recopilando indicadores técnicos de {nombre_empresa} de Alpha Vantage…",
        f"Recopilando métricas derivadas de opciones de {nombre_empresa} de Volvue…",
        f"Extrayendo noticias recientes de {nombre_empresa} de The Guardian…",
        f"Extrayendo noticias recientes de {nombre_empresa} de The New York Times…",
        f"Extrayendo noticias recientes de {nombre_empresa} de EODHD…",
        "Analizando el sentimiento de las noticias recopiladas con LLM…",
        "Fusionando métricas técnicas con análisis de sentimiento…",
        "Ejecutando el modelo de predicción financiera…",
    ]

    # ---- Etapa 1: técnicos ----
    msg.markdown(f"### {etapas[0]}")
    os.makedirs(os.path.dirname(MASTER) or ".", exist_ok=True)
    cb = make_stage_progress(progress, msg, 0, 15)
    try:
        res = data.update_master_technicals(
            master_csv_path=MASTER,
            ticker=ticker,
            on_progress=cb,
            write_back=True,
            years_hist=10,
        )
    except Exception as e:
        st.error(f"Error en técnicos: {e}")
        st.session_state.running = False
        st.stop()

    # ---- Etapa 2: opciones ----
    msg.markdown(f"### {etapas[1]}")
    cb2 = make_stage_progress(progress, msg, 15, 30)
    try:
        res_opt = data.update_master_options_from_csvs(
            master_csv_path=MASTER,
            ticker=ticker,
            on_progress=cb2,
            write_back=True,
        )
    except Exception as e:
        st.error(f"Error en opciones: {e}")
        st.session_state.running = False
        st.stop()

    # ---- Etapa 3: noticias ----
    cb3 = make_stage_progress(progress, msg, 30, 50)
    news_from, news_to = _compute_news_range(MASTER, fallback_days=60)
    dt_from, dt_to = str(news_from), str(news_to)

    raw_csvs = []
    try:
        msg.markdown(f"### {etapas[2]}") 
        cb3(0, "") 
        g_meta, g_err = _run_news_script(
            _script_path("theguardian.py"),
            ["--from", dt_from, "--to", dt_to, "--out-dir", "data/master"],
            progress_cb=None
        )
        if g_meta and g_meta.get("csv"): raw_csvs.append(g_meta["csv"])

        msg.markdown(f"### {etapas[3]}")
        cb3(33, "")
        nyt_script = "thenewyorktimes.py"
        if not os.path.exists(_script_path(nyt_script)):
            nyt_script = "newyorktimes.py"
        n_meta, n_err = _run_news_script(
            _script_path(nyt_script),
            ["--from", dt_from, "--to", dt_to, "--out-dir", "data/master"],
            progress_cb=None
        )
        if n_meta and n_meta.get("csv"): raw_csvs.append(n_meta["csv"])

        msg.markdown(f"### {etapas[4]}")
        cb3(66, "")
        if os.path.exists(_script_path("eodhd.py")):
            e_meta, e_err = _run_news_script(
                _script_path("eodhd.py"),
                ["--from", dt_from, "--to", dt_to, "--out-dir", "data/master"],
                progress_cb=None
            )
            if e_meta and e_meta.get("csv"): raw_csvs.append(e_meta["csv"])
        else:
            cb3(75, "")

        cb3(90, "")
        unified_csv = _unify_news(raw_csvs, "data/master", dt_from, dt_to)
        cb3(100, "")
        news_res = {"raw_files": raw_csvs, "unified_csv": unified_csv}

    except Exception as e:
        st.error(f"Error recopilando/unificando noticias: {e}")
        st.session_state.running = False
        st.stop()

    # ---- Etapa 4: LLM ----
    msg.markdown(f"### {etapas[5]}")
    cb4 = make_stage_progress(progress, msg, 50, 75)
    try:
        try:
            llm_res = sentiment.classify_and_score_unified_news(
                unified_csv_path=news_res.get("unified_csv"),
                ticker=ticker,
                on_progress=cb4,
                master_csv=MASTER,
            )
        except TypeError:
            llm_res = sentiment.classify_and_score_unified_news(
                unified_csv_path=news_res.get("unified_csv"),
                ticker=ticker,
                on_progress=cb4,
            )
    except Exception as e:
        st.error(f"Error evaluando relevancia/sentimiento: {e}")
        st.session_state.running = False
        st.stop()

    # ---- Etapa 5: escribir maestro ----
    msg.markdown(f"### {etapas[6]}")
    cb5 = make_stage_progress(progress, msg, 75, 90)
    try:
        upd = sentiment.update_master_with_daily_sentiment(
            master_csv_path=MASTER,
            daily_mean_csv=llm_res.get("daily_mean_csv"),
            on_progress=cb5
        )
        n_borrados = _limpiar_archivos_temporales("data/master")
    except Exception as e:
        st.error(f"Error actualizando el maestro con sentimiento: {e}")
        st.session_state.running = False
        st.stop()

    # ---- Etapa 6: inferencia ----
    msg.markdown(f"### {etapas[7]}")
    hkey = HKMAP[horizonte]
    try:
        res_inf = infer.run_inference(hkey, maestro_path=MASTER, use_llm=True)
    except Exception as e:
        st.error(f"Error en inferencia: {e}")
        st.session_state.running = False
        st.stop()

    for p in range(90, 100):
        progress.progress(p + 1)
        time.sleep(0.01)

    dias = BDAY_MAP[horizonte]
    target_date = (Timestamp.today().normalize() + BDay(dias)).date()

    st.session_state.running = False
    st.session_state.done = True
    st.session_state.result = {
        "ticker": ticker, "horizonte": horizonte, "dias": dias,
        "target_date": fmt(target_date),
        "infer": res_inf,
    }
    st.rerun()

with zona_resultados.container():
    if st.session_state.done:
        r = st.session_state.result
        res_inf = r.get("infer") or {}
        
        pred = res_inf.get("pred_pct", 0.0)
        ui_val = res_inf.get("ui_pct", pred) 
        
        narrative_raw = res_inf.get("narrative") or "No hay narrativa disponible."
        
        if st.session_state.simple_text:
            narrative_to_show = st.session_state.simple_text
            is_simplified = True
        else:
            narrative_to_show = narrative_raw
            is_simplified = False

        texto_horizonte = r.get("horizonte", "—")

        if texto_horizonte == "1 día":
            color = "#fbbf24"
            flecha = "▲" if pred >= 0 else "▼"
        elif pred >= 0:
            color = "#4ade80"
            flecha = "▲"
        else:
            color = "#f87171"
            flecha = "▼"

        etiqueta_completa = f"PRONÓSTICO A {texto_horizonte}"

        html_headline = (
            f'<div style="display: flex; flex-direction: row; align-items: center; justify-content: flex-end; margin-top: 10px; margin-bottom: 20px;">'
            f'<span style="font-family: sans-serif; font-size: 20px; color: #ffffff; font-weight: 600; text-transform: uppercase; margin-right: 15px; white-space: nowrap;">'
            f'{etiqueta_completa}'
            f'</span>'
            f'<span style="font-family: sans-serif; font-size: 45px; font-weight: 700; color: {color}; line-height: 1; white-space: nowrap;">'
            f'{flecha}{ui_val}%'
            f'</span>'
            f'</div>'
        )
        st.markdown(html_headline, unsafe_allow_html=True)

        bg_color = "rgba(20, 83, 45, 0.2)" if is_simplified else "rgba(28, 131, 225, 0.1)"
        border_color = "rgba(20, 83, 45, 0.5)" if is_simplified else "rgba(28, 131, 225, 0.4)"

        # Ajuste de párrafos: 4px de separación (o lo que quieras ajustar)
        texto_html = narrative_to_show.replace("\n", "<div style='height: 4px;'></div>")

        html_box = f"""
        <div style="
            background-color: {bg_color}; 
            border: 1px solid {border_color};
            border-radius: 9px;
            padding: 16px;
            color: #cbd5e1; 
            font-family: 'Source Sans Pro', sans-serif;
            font-size: 16px;
            line-height: 1.5;
        ">
            {texto_html}
        </div>
        """
        st.markdown(html_box, unsafe_allow_html=True)

        # --- BOTÓN DE SIMPLIFICAR ---
        if not is_simplified:
            # MARGEN NEGATIVO: Esto es lo que sube el botón hacia arriba
            st.markdown("<div style='margin-top: -20px;'></div>", unsafe_allow_html=True)

            col_spacer, col_btn = st.columns([3, 1])
            with col_btn:
                btn_placeholder = st.empty()
                
                # Botón VERDE (primary)
                clicked = btn_placeholder.button("🧠 Explicación sencilla", type="primary", use_container_width=True)
                
                if clicked:
                    bar = btn_placeholder.progress(0)
                    for i in range(100):
                        time.sleep(0.005) 
                        bar.progress(i + 1)
                    
                    simple = _generar_explicacion_simple(narrative_raw)
                    st.session_state.simple_text = simple
                    st.rerun()
