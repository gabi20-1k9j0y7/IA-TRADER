#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eodhd.py — Extracción EODHD → AMZN.US (exacto al entrenamiento en filtros)
- Título debe contener 'Amazon' (palabra completa).
- Excluye PR wires, clickbait y entretenimiento/Prime Video.
- Excluye contexto 'Amazon rainforest' salvo señales corporativas.
- Genera resumen extractivo y guarda CSV estándar:
  [Fecha, Titular, Resumen, URL, Seccion, Desk, Tipo, Autor, Fuente]
"""
import os, sys, re, json, time, argparse
from datetime import timezone, timedelta
from urllib.parse import urlparse
import pandas as pd
import requests

TZ_ET = timezone(timedelta(hours=-5))
BASE_URL = "https://eodhd.com/api/news"
TICKER   = "AMZN.US"
PER_PAGE = 1000
SLEEP_S  = 0.5

def progress(pct: int, msg: str):
    sys.stderr.write(f"%%{pct}|{msg}\n"); sys.stderr.flush()

def get_api_key():
    try:
        import streamlit as st  # type: ignore
        secrets = dict(st.secrets) if hasattr(st, "secrets") else {}
    except Exception:
        secrets = {}
    return secrets.get("EODHD_API_KEY") or os.getenv("EODHD_API_KEY") or "DEMO"

PR_DOMAINS = [
    "globenewswire","prnewswire","businesswire","accesswire",
    "newsfilecorp","newsdirect","einnews","newswire","prweb",
    "/press-releases","/pressrelease","/press-release"
]
CLICKBAIT_PATTERNS = [
    r"\bwhat to know\b", r"\bwhat we know\b", r"\beverything you need to know\b",
    r"\byou won'?t believe\b", r"\bmust[- ]see\b", r"\bbreaking\b", r"\bgoes viral\b",
    r"\btop\s?\d+\b", r"\bbest\b", r"\bhow to watch\b", r"\bwhere to watch\b",
    r"\bstreaming guide\b", r"\brelease date\b", r"\btrailer\b", r"\brecap\b", r"\breview\b",
]
ENTERTAINMENT_PATTERNS = [
    r"\bprime video\b", r"\bseries?\b", r"\bseason\b", r"\bepisode\b",
    r"\bmr\.?\s*&?\s*mrs\.?\s*smith\b", r"\brings?\s*of\s*power\b", r"\breacher\b",
    r"\bouter range\b", r"\bthe boys\b", r"\bexpats\b", r"\bmy lady jane\b", r"\bcast\b"
]
RAINFOREST_BAD = [
    r"\brainforest\b", r"\bdeforestation\b", r"\bdrought\b", r"\bamazon river\b",
    r"\bindigenous\b", r"\btribes?\b"
]
CORP_MARKERS = [r"\bamazon\.com\b", r"\baws\b", r"\bamazon web services\b", r"\bnasdaq:\s*amzn\b"]

def has_pr_source(link: str, source: str) -> bool:
    s = f"{link} {source}".lower()
    return any(dom in s for dom in PR_DOMAINS)

def title_has_amazon(title: str) -> bool:
    return isinstance(title, str) and bool(re.search(r"\bamazon\b", title, flags=re.I))

def contains_focus_terms(text: str) -> bool:
    if not isinstance(text, str): return False
    t = str(text).lower()
    return ("amazon" in t) or ("aws" in t)

def norm_title(t: str) -> str:
    if not isinstance(t, str): return ""
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9 ]+", "", t)
    return t.strip()

def is_clickbaity(title: str) -> bool:
    if not isinstance(title, str): return False
    t = title.lower()
    short = len(t) < 28
    bad = any(re.search(p, t) for p in CLICKBAIT_PATTERNS)
    ent = any(re.search(p, t) for p in ENTERTAINMENT_PATTERNS)
    return bad or ent or short

def rainforest_context(text: str) -> bool:
    if not isinstance(text, str): return False
    t = text.lower()
    has_rain = any(re.search(p, t) for p in RAINFOREST_BAD)
    has_corp = any(re.search(p, t) for p in CORP_MARKERS)
    return has_rain and not has_corp

def domain_from_link(link: str) -> str:
    try: return urlparse(link).netloc.lower()
    except Exception: return ""

def summarize_extract(text: str, max_chars=420) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:max_chars]

def fetch_eodhd(dt_from: str, dt_to: str, api_token: str):
    all_rows, offset = [], 0
    while True:
        params = {
            "s": TICKER, "from": dt_from, "to": dt_to,
            "limit": PER_PAGE, "offset": offset, "fmt": "json",
            "api_token": api_token
        }
        r = requests.get(BASE_URL, params=params, timeout=30)
        if r.status_code != 200:
            break
        txt = r.text.strip()
        page = json.loads(txt) if txt else []
        if not page:
            break
        all_rows.extend(page)
        if len(page) < PER_PAGE:
            break
        offset += PER_PAGE
        time.sleep(SLEEP_S)
    return all_rows

def main():
    ap = argparse.ArgumentParser("EODHD fetcher")
    ap.add_argument("--from", dest="dt_from", required=True)
    ap.add_argument("--to", dest="dt_to", required=True)
    ap.add_argument("--out-dir", dest="out_dir", required=True)
    args = ap.parse_args()

    fr = pd.to_datetime(args.dt_from); to = pd.to_datetime(args.dt_to)
    os.makedirs(args.out_dir, exist_ok=True)

    api = get_api_key()
    progress(5, "EODHD: consultando…")
    try:
        rows = fetch_eodhd(fr.date().isoformat(), to.date().isoformat(), api)
    except Exception as e:
        progress(100, f"EODHD error HTTP: {e}")
        rows = []

    # Flatten
    def flatten(row: dict):
        sent = row.get("sentiment") or {}
        symbols = row.get("symbols") or []
        if isinstance(symbols, str):
            symbols = [x.strip() for x in symbols.split(",") if x.strip()]
        return {
            "date": row.get("date"),
            "title": row.get("title"),
            "content": row.get("content"),
            "link": row.get("link"),
            "source": row.get("source"),
            "lang": row.get("lang"),
            "symbols": ",".join(symbols),
            "tags": ",".join(row.get("tags") or []),
            "polarity": sent.get("polarity"),
            "neg": sent.get("neg"),
            "neu": sent.get("neu"),
            "pos": sent.get("pos"),
        }

    df_raw = pd.DataFrame([flatten(x) for x in rows])
    for col in ["title","content","link","source","symbols","tags","lang"]:
        if col not in df_raw.columns: df_raw[col] = ""

    df_raw["symbols_list"]  = df_raw["symbols"].apply(lambda s: [x.strip() for x in str(s).split(",") if x.strip()])
    df_raw["symbols_count"] = df_raw["symbols_list"].apply(len)
    df_raw["has_amzn"]      = df_raw["symbols_list"].apply(lambda xs: "AMZN.US" in xs)
    df_raw["title_norm"]    = df_raw["title"].apply(norm_title)
    df_raw["content_len"]   = df_raw["content"].map(lambda x: len(str(x)))
    df_raw["source_domain"] = df_raw["link"].map(domain_from_link)

    mask = True
    mask = mask & df_raw["title"].apply(title_has_amazon)
    mask = mask & ~df_raw.apply(lambda r: has_pr_source(r["link"], r["source"]), axis=1)
    mask = mask & df_raw["has_amzn"] & (df_raw["symbols_count"] <= 3)
    mask = mask & (df_raw["content_len"] >= 300)
    mask = mask & (df_raw["title"].apply(lambda t: not is_clickbaity(t)))
    mask = mask & (df_raw["content"].apply(lambda t: not rainforest_context(t)))
    mask = mask & (df_raw["content"].apply(contains_focus_terms) | (df_raw["content_len"] == 0))

    df_f = df_raw[mask].copy()
    df_f = df_f.sort_values("date").drop_duplicates(subset=["link"], keep="first")
    df_f = df_f.sort_values("date").drop_duplicates(subset=["title_norm"], keep="first")

    out_rows = []
    for _, r in df_f.iterrows():
        dtv = pd.to_datetime(r.get("date"), errors="coerce", utc=True)
        if pd.isna(dtv): 
            continue
        out_rows.append({
            "Fecha": dtv.tz_convert(TZ_ET).date(),
            "Titular": (r.get("title") or "").strip(),
            "Resumen": summarize_extract(r.get("content") or ""),
            "URL": r.get("link") or "",
            "Seccion": r.get("source_domain") or "",
            "Desk": "",
            "Tipo": "Article",
            "Autor": "",
            "Fuente": "EODHD",
        })

    out_path = os.path.join(args.out_dir, f"eodhd_{fr.date()}_{to.date()}.csv")
    if out_rows:
        pd.DataFrame(out_rows).to_csv(out_path, index=False, encoding="utf-8")
    else:
        cols = ["Fecha","Titular","Resumen","URL","Seccion","Desk","Tipo","Autor","Fuente"]
        pd.DataFrame(columns=cols).to_csv(out_path, index=False, encoding="utf-8")

    progress(100, f"EODHD listo: {len(out_rows)} filas")
    print(json.dumps({"csv": out_path, "count": int(len(out_rows)), "source": "EODHD",
                      "from": str(fr.date()), "to": str(to.date())}))
    sys.exit(0)

if __name__ == "__main__":
    main()
