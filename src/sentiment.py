#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sentiment.py — Relevancia → Puntuación (solo relevantes) → Media diaria
- La media diaria cubre desde el inicio del intervalo de noticias HASTA HOY.
- Si faltan días en medio, se rellenan con 0.
- Salidas:
  1) "<input_stem> - con Relevante.csv"
  2) "<input_stem> - puntuado.csv"                 (SOLO relevantes)
  3) "<input_stem> - puntuado - diario_media.csv"  (exacto: Fecha, Media_Puntuacion; rango completo hasta hoy)

Requisitos: pip install openai pandas
Lee la API key de OPENAI_API_KEY (env o st.secrets).
"""
import os, re, json, time, hashlib, sys, argparse
from pathlib import Path
from typing import Optional, Callable, Dict, Any

import pandas as pd

# ======= OpenAI client =======
def _get_openai_client(explicit_key: Optional[str] = None):
    key = explicit_key
    if not key:
        try:
            import streamlit as st  # type: ignore
            if hasattr(st, "secrets"):
                key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("openai_api_key")
        except Exception:
            pass
    key = key or os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
    if not key:
        raise RuntimeError("Falta OPENAI_API_KEY (en st.secrets o variable de entorno).")
    from openai import OpenAI  # lazy import
    return OpenAI(api_key=key)

# ======= Config y prompts =======
MODEL_RELEVANCE = "gpt-4o-mini"
MODEL_SCORE     = "gpt-4o-mini"
MAX_OUTPUT_TOKENS_RELEV = 8
MAX_OUTPUT_TOKENS_SCORE = 8
SAVE_EVERY = 500

SYSTEM_PROMPT_RELEV = """
You are a finance/news relevance classifier. Decide if a news item is relevant to Amazon.com, Inc.’s stock (AMZN).
Output EXACTLY one JSON: {"Relevante":"Si"} or {"Relevante":"No"} — in Spanish, initial capital S/N, no extra fields.

Label “Si” if it plausibly moves AMZN or reflects material fundamentals/risks:
- Earnings, revenue, guidance, KPIs (AWS growth/margins), buybacks/dividends.
- Major AWS events (outages, security, pricing, large client wins/losses, AI/infra with clear scale).
- M&A/divestitures/investments (e.g., Project Kuiper milestones with commercial impact).
- Government/regulatory/legal actions materially involving Amazon (FTC/DoJ/EU, big fines, union rulings, targeted taxation).
- Leadership/board changes, large layoffs/hiring waves, strategy shifts.
- Prime/retail logistics with clear financial impact (Prime price change, nationwide strikes, broad delivery disruptions).
- Large partnerships/contracts with quantified or clearly material scope.

Label “No” if:
- It is about the Amazon rainforest/region (“Amazonas”, “Amazonía”, “Manaus”, deforestation, indigenous communities…).
- Local/minor items (single warehouse accidents, local charity/store opening) without broad financial impact.
- Clickbait, listicles, coupons, how-to guides, product reviews, marketing fluff, celebrity gossip.
- Generic industry news without specific material link to Amazon.
- Duplicate/trivial updates with no new material info.
- Missing summary and vague title with no material signal.

Be conservative: default to “No” unless materiality is clear.

INPUT:
Title: <string>
Summary: <string or empty>

OUTPUT (strict):
{"Relevante":"Si"}  OR  {"Relevante":"No"}
""".strip()

SYSTEM_PROMPT_SCORE = """
You are a financial impact rater for Amazon.com, Inc. (AMZN).
Return ONLY one integer from -10 to +10 (no words). Positive = good for AMZN; negative = bad.
Bigger |score| = more material (magnitude, breadth/duration, directness to AMZN/AWS/Prime/Whole Foods/Kuiper, novelty, credibility).

Scale (guidance):
+9..+10 blockbuster beat/guidance raise; multi-billion contract; antitrust case dismissed; big buyback/dividend.
+6..+8  clear beat; AWS acceleration; major partnership/pricing with numbers; accretive M&A.
+3..+5  moderate positive; some numbers/scope.
+1..+2  mild/uncertain positive.
 0       neutral/ambiguous; rainforest/region “Amazonas/Amazonía/Manaus”.
-1..-2  mild/uncertain negative.
-3..-5  miss/guidance trim; sizable fine; regional strike; slowdown datapoint.
-6..-8  major negative: AWS outage/security; antitrust/regulatory action; broad labor disruption; big breach; guidance cut.
-9..-10 severe negative: multi-region/multi-day outage; lost landmark case with heavy remedies; breakup-like remedy; huge recurring costs/taxation.
""".strip()

# ======= Prefiltros =======
RE_RAINFOREST = re.compile(r"\b(amazonas|amazonía|amazônia|manaus|rainforest|amazon basin|amaz[oó]n.+forest)\b", re.I)
RE_CLICKBAIT  = re.compile(r"\b(how to|guide|coupon|deal|promo|trick|tips|hacks|ranking|top \d+|you won.?t believe)\b", re.I)
RE_LOCAL      = re.compile(r"\b(local|community|neighborhood|parish|county fair|charity 5k|school fundraiser)\b", re.I)
RE_COMPANY    = re.compile(
    r"\b(amazon\.com|amzn|amazon|aws|prime( video)?|whole foods|ring|twitch|kindle|zoox|irobot|mgm|audible|kuiper|project kuiper|amazon go|amazon fresh)\b",
    re.I,
)

def cheap_prefilter(title: str, summary: str) -> Optional[str]:
    t = (title or "").strip()
    s = (summary or "").strip()
    txt = f"{t} {s}"
    if RE_RAINFOREST.search(txt): return "No"
    if RE_CLICKBAIT.search(txt):  return "No"
    if RE_LOCAL.search(txt):      return "No"
    if not RE_COMPANY.search(txt): return "No"
    if not t and not s:           return "No"
    return None

# ======= Helpers =======
def key_for(title: str, summary: str) -> str:
    return hashlib.sha256(f"{title}||{summary}".encode("utf-8")).hexdigest()

def read_csv_any(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, engine="python")

def _progress(cb: Optional[Callable[[int,str], None]], pct: int, msg: str):
    if cb:
        try: cb(pct, msg)
        except Exception: pass
  

def _range_from_filename(p: Path):
    """Intenta extraer _YYYY-MM-DD_YYYY-MM-DD del nombre del fichero."""
    m = re.search(r"_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})", p.name)
    if not m:
        return None, None
    start = pd.to_datetime(m.group(1)).date()
    end   = pd.to_datetime(m.group(2)).date()
    return start, end

# ======= Paso 1: relevancia =======
def mark_relevance(in_csv: Path, out_csv: Path, api_key: Optional[str] = None,
                   on_progress: Optional[Callable[[int,str], None]] = None) -> Dict[str, Any]:
    client = _get_openai_client(api_key)
    df = read_csv_any(in_csv).copy()

    col_titular = "Titular" if "Titular" in df.columns else df.columns[1]
    col_resumen = "Resumen" if "Resumen" in df.columns else (df.columns[2] if len(df.columns) > 2 else None)
    if col_resumen is None:
        df["Resumen"] = ""
        col_resumen = "Resumen"

    if "Relevante" not in df.columns:
        df["Relevante"] = ""

    cache: Dict[str, str] = {}
    n = len(df)

    for i, row in df.iterrows():
        if df.at[i, "Relevante"]:
            continue
        title = str(row.get(col_titular, "") or "")
        summ  = str(row.get(col_resumen, "") or "")

        pre = cheap_prefilter(title, summ)
        if pre is not None:
            df.at[i, "Relevante"] = pre
        else:
            k = key_for(title, summ)
            if k in cache:
                df.at[i, "Relevante"] = cache[k]
            else:
                try:
                    resp = client.chat.completions.create(
                        model=MODEL_RELEVANCE,
                        temperature=0,
                        max_tokens=MAX_OUTPUT_TOKENS_RELEV,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT_RELEV},
                            {"role": "user", "content": f"Title: {title}\nSummary: {summ}"},
                        ],
                    )
                    content = (resp.choices[0].message.content or "").strip()
                    data = json.loads(content) if content.startswith("{") else {}
                    label = "Si" if data.get("Relevante") == "Si" else "No"
                except Exception:
                    label = "No"
                cache[k] = label
                df.at[i, "Relevante"] = label

        if (i + 1) % SAVE_EVERY == 0:
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            _progress(on_progress, int(5 + 45*(i+1)/max(1,n)), f"Relevancia: {i+1}/{n} (guardado parcial)")

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    si = int((df["Relevante"] == "Si").sum())
    no = int((df["Relevante"] == "No").sum())
    _progress(on_progress, 50, f"Relevancia lista: Si={si} No={no}")
    return {"relevant_yes": si, "relevant_no": no, "out_csv": str(out_csv)}

# ======= Paso 2: puntuación (SOLO relevantes) =======
def score_news(in_csv: Path, out_csv: Path, api_key: Optional[str] = None,
               on_progress: Optional[Callable[[int,str], None]] = None) -> Dict[str, Any]:
    client = _get_openai_client(api_key)
    df_all = read_csv_any(in_csv).copy()

    # Filtra SOLO relevantes "Si"
    if "Relevante" in df_all.columns:
        df = df_all[df_all["Relevante"].astype(str).str.strip().eq("Si")].copy()
    else:
        df = df_all.copy()

    col_titular = "Titular" if "Titular" in df.columns else (df.columns[1] if len(df.columns)>1 else None)
    col_resumen = "Resumen" if "Resumen" in df.columns else (df.columns[2] if len(df.columns) > 2 else None)
    if col_titular is None:
        raise ValueError("No se encontró la columna de Titular.")
    if col_resumen is None:
        df["Resumen"] = ""
        col_resumen = "Resumen"

    if df.shape[0] == 0:
        # CSV vacío pero con cabeceras
        cols = list(df_all.columns)
        if "Puntuacion" not in cols:
            cols.append("Puntuacion")
        pd.DataFrame(columns=cols).to_csv(out_csv, index=False, encoding="utf-8-sig")
        _progress(on_progress, 90, "No hay relevantes. 'puntuado.csv' vacío.")
        return {"out_csv": str(out_csv), "tokens_prompt": 0, "tokens_completion": 0}

    if "Puntuacion" not in df.columns:
        df["Puntuacion"] = ""

    n = len(df)
    for i, row in df.iterrows():
        title = str(row.get(col_titular, "") or "")
        summ  = str(row.get(col_resumen, "") or "")
        try:
            resp = client.chat.completions.create(
                model=MODEL_SCORE,
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS_SCORE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_SCORE},
                    {"role": "user", "content": f"Title: {title}\nSummary: {summ}\nAnswer with a single integer between -10 and 10. Return only the number."},
                ],
            )
            content = (resp.choices[0].message.content or "").strip()
            m = re.search(r"-?\d+", content)
            val = int(m.group()) if m else 0
            val = max(-10, min(10, val))
        except Exception:
            val = 0
        df.at[i, "Puntuacion"] = val

        if (i + 1) % 50 == 0:
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            _progress(on_progress, int(50 + 40*(i+1)/max(1,n)), f"Puntuando (relevantes): {i+1}/{n}")

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    _progress(on_progress, 90, f"Puntuación lista (relevantes={n}).")
    return {"out_csv": str(out_csv)}

# ======= Paso 3: media diaria (rango completo hasta HOY; 2 columnas) =======
def daily_mean(in_scored_csv: Path, out_daily_csv: Path,
               on_progress: Optional[Callable[[int,str], None]] = None) -> Dict[str, Any]:
    df = read_csv_any(in_scored_csv).copy()

    # Columnas esperadas
    col_fecha = "Fecha" if "Fecha" in df.columns else df.columns[0]
    if "Puntuacion" not in df.columns:
        # Sin puntuadas → diario vacío
        out = pd.DataFrame(columns=["Fecha", "Media_Puntuacion"])
        out.to_csv(out_daily_csv, index=False, encoding="utf-8-sig")
        if on_progress: on_progress(100, "Media diaria lista: 0 días con noticia.")
        return {"daily_mean_csv": str(out_daily_csv), "rows": 0}

    # Parse de fecha MUY robusto (primero yearfirst, luego dayfirst si hiciera falta)
    dt = pd.to_datetime(df[col_fecha], errors="coerce", yearfirst=True)
    if dt.notna().sum() == 0:
        dt = pd.to_datetime(df[col_fecha], errors="coerce", dayfirst=True)

    pts  = pd.to_numeric(df["Puntuacion"], errors="coerce")
    mask = dt.notna() & pts.notna()

    # SOLO días con noticia (nada de reindex ni ceros)
    s = pts[mask].groupby(dt[mask].dt.normalize()).mean().sort_index()

    out = pd.DataFrame({
        "Fecha": s.index.strftime("%Y-%m-%d"),    # ISO para no liar day/month
        "Media_Puntuacion": s.round(3).values
    })
    out.to_csv(out_daily_csv, index=False, encoding="utf-8-sig")
    if on_progress: 
        on_progress(100, f"Media diaria lista: {len(out)} días con noticia.")
    return {"daily_mean_csv": str(out_daily_csv), "rows": int(len(out))}

def update_master_with_daily_sentiment(master_csv_path: str,
                                         daily_mean_csv: str,
                                         on_progress=None) -> dict:
    import pandas as pd

    def _cb(p, m):
        if on_progress:
            try: on_progress(p, m)
            except Exception: pass

    _cb(0, "Leyendo maestro…")
    master_txt = pd.read_csv(master_csv_path, dtype=str, keep_default_na=False)
    if "Date" not in master_txt.columns:
        raise ValueError("El maestro no tiene columna 'Date'.")

    # Parseo robusto de fechas del maestro
    dt1 = pd.to_datetime(master_txt["Date"], format="%d-%m-%y", errors="coerce")
    dt2 = pd.to_datetime(master_txt.loc[dt1.isna(), "Date"], errors="coerce", dayfirst=True)
    date_master = dt1.fillna(dt2).dt.normalize()

    _cb(10, "Leyendo diario_media…")
    daily = pd.read_csv(daily_mean_csv)
    date_col = "Fecha" if "Fecha" in daily.columns else ("Date" if "Date" in daily.columns else None)
    val_col  = "Media_Puntuacion" if "Media_Puntuacion" in daily.columns else None
    if not date_col or not val_col:
        raise ValueError("El CSV diario debe tener columnas 'Fecha' y 'Media_Puntuacion'.")

    # Asegura que las fechas de noticias se parsean correctamente (YYYY-MM-DD)
    dt_news = pd.to_datetime(daily[date_col], errors="coerce", yearfirst=True).dt.normalize()
    s_news = pd.to_numeric(daily[val_col], errors="coerce").groupby(dt_news).mean().sort_index()
    if s_news.empty:
        _cb(100, "No hay datos en el diario_media; maestro sin cambios.")
        return {"updated_rows": 0, "missing_to_zero": 0, "last_sent_before": None, "last_sent_after": None}

    # --- INICIO DE LA LÓGICA CORREGIDA ---

    col_news = "news_sent_mean"
    if col_news not in master_txt.columns:
        master_txt[col_news] = "" # pd.to_numeric lo convertirá en NaN

    # 1. Empezar con la columna existente de puntuaciones
    final_col = pd.to_numeric(master_txt[col_news], errors="coerce")

    # 2. Mapear las *nuevas* puntuaciones de s_news al índice del maestro
    map_new_scores = date_master.map(s_news)

    # 3. Identificar filas que tienen una puntuación *nueva* en diario_media
    mask_has_new_score = map_new_scores.notna()
    
    # 4. Escribir esas puntuaciones nuevas (Regla 1).
    #    Esto sobrescribe CUALQUIER valor (antiguo, 0, o NaN)
    final_col.loc[mask_has_new_score] = map_new_scores.loc[mask_has_new_score]

    # 5. Definir el rango para rellenar con ceros
    #    Buscamos la PRIMERA fecha en el maestro que AÚN esté vacía (NaN)
    #    después de haber escrito las puntuaciones nuevas.
    mask_is_still_na = final_col.isna()
    start_write_zeros = None
    
    if mask_is_still_na.any():
        # .idxmax() encuentra el índice de la primera fila True (o sea, el primer NaN)
        first_nan_index = mask_is_still_na.idxmax()
        start_write_zeros = date_master.loc[first_nan_index]

    today = pd.Timestamp.today().normalize()
    
    # 6. Identificar filas que necesitan un CERO (Regla 2):
    mask_needs_zero = pd.Series(False, index=master_txt.index) # Por defecto, no rellenar nada
    
    if start_write_zeros is not None:
        # El rango de relleno va desde el primer hueco que encontramos hasta hoy
        mask_zero_fill_range = date_master.notna() & (date_master >= start_write_zeros) & (date_master <= today)
        
        # Rellenamos CON CERO si:
        # - Está en el rango de relleno (desde el primer hueco hasta hoy)
        # - NO vino un dato nuevo del 'diario_media' (porque si no, ya se habría escrito)
        # - Y (por seguridad) sigue estando vacío
        mask_needs_zero = mask_zero_fill_range & map_new_scores.isna() & final_col.isna()

    # 7. Rellenar solo esas filas con 0.0 (Regla 3: no toca datos antiguos)
    final_col.loc[mask_needs_zero] = 0.0

    # 8. Asignar la columna final actualizada al dataframe
    master_txt[col_news] = final_col

    # --- FIN DE LA LÓGICA CORREGIDA ---

    _cb(70, "Calculando derivadas (w3, w21, w63, w126)…")
    s_full = pd.to_numeric(master_txt[col_news], errors="coerce")
    # min_periods=1 → no deja huecos dentro del tramo
    for w in (3, 21, 63, 126):
        master_txt[f"news_mean_w{w}"] = s_full.rolling(window=w, min_periods=1).mean()
        master_txt[f"news_std_w{w}"]  = s_full.rolling(window=w, min_periods=1).std(ddof=0)

    # Restaurar formato de fecha del maestro
    fmt_date = date_master.dt.strftime("%d-%m-%y")
    master_txt["Date"] = master_txt["Date"].where(date_master.isna(), fmt_date)

    _cb(90, "Guardando maestro…")
    master_txt.to_csv(master_csv_path, index=False, encoding="utf-8-sig")
    _cb(100, "Maestro actualizado.")

    return {
        "updated_rows": int((mask_has_new_score | mask_needs_zero).sum()),
        "missing_to_zero": int(mask_needs_zero.sum())
    }

# ======= Orquestador =======
def classify_and_score_unified_news(unified_csv_path: str, ticker: str = "AMZN",
                                    on_progress: Optional[Callable[[int,str], None]] = None,
                                    api_key: Optional[str] = None) -> Dict[str, Any]:
    p = Path(unified_csv_path)
    stem = p.stem
    out_dir = p.parent

    relev_csv = out_dir / f"{stem} - con Relevante.csv"
    scored_csv = out_dir / f"{stem} - puntuado.csv"
    daily_csv  = out_dir / f"{stem} - puntuado - diario_media.csv"

    r1 = mark_relevance(p, relev_csv, api_key=api_key, on_progress=on_progress)
    r2 = score_news(relev_csv, scored_csv, api_key=api_key, on_progress=on_progress)   # SOLO relevantes
    r3 = daily_mean(scored_csv, daily_csv, on_progress=on_progress)                    # Rango completo hasta HOY

    return {
        "relevant_csv": str(relev_csv),
        "unified_v3_csv": str(relev_csv),
        "unified_scored_csv": str(scored_csv),
        "daily_mean_csv": str(daily_csv),
        **r1, **r2, **r3
    }

# ======= CLI =======
def _run_cli():
    ap = argparse.ArgumentParser("Relevancia → Puntuación → Media diaria (rango hasta HOY)")
    ap.add_argument("--input", required=True, help="CSV unificado (NOTICIAS_UNIFICADAS_*.csv)")
    ap.add_argument("--api-key", default=None, help="Opcional: API key si no usas OPENAI_API_KEY")
    args = ap.parse_args()

    try:
        res = classify_and_score_unified_news(args.input, ticker="AMZN", on_progress=None, api_key=args.api_key)
        print(json.dumps(res, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    _run_cli()
