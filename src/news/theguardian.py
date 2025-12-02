#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
theguardian.py — Descarga noticias de The Guardian sobre Amazon
Salida: CSV con columnas [Fecha, Titular, Resumen, URL, Seccion, Desk, Tipo, Autor, Fuente]
Progreso por stderr: líneas "%%<pct>|<mensaje>"
Salida final por stdout: JSON {"csv":..., "count":..., "source":"Guardian","from":"YYYY-MM-DD","to":"YYYY-MM-DD"}
"""
import os, sys, re, json, time, argparse
from datetime import timezone, timedelta
from urllib.parse import urlparse
import pandas as pd
import requests

TZ_ET = timezone(timedelta(hours=-5))

def progress(pct: int, msg: str):
    sys.stderr.write(f"%%{pct}|{msg}\n"); sys.stderr.flush()

def get_api_key():
    # st.secrets primero; si no, variable de entorno GUARDIAN_API_KEY
    try:
        import streamlit as st  # type: ignore
        secrets = dict(st.secrets) if hasattr(st, "secrets") else {}
    except Exception:
        secrets = {}
    return secrets.get("GUARDIAN_API_KEY") or os.getenv("GUARDIAN_API_KEY")

def title_has_amazon(title: str) -> bool:
    return isinstance(title, str) and bool(re.search(r"\bamazon\b", title, re.I))

def domain_from_link(link: str) -> str:
    try: return urlparse(link).netloc.lower()
    except Exception: return ""

def summarize(text: str, max_chars=420) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:max_chars]

def norm_title(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9 ]+", "", t)
    return t.strip()

def main():
    ap = argparse.ArgumentParser("Guardian fetcher")
    ap.add_argument("--from", dest="dt_from", required=True)
    ap.add_argument("--to", dest="dt_to", required=True)
    ap.add_argument("--out-dir", dest="out_dir", required=True)
    args = ap.parse_args()

    api_key = get_api_key()
    if not api_key:
        print(json.dumps({"error": "missing_guardian_api_key"}))
        sys.exit(1)

    fr, to = pd.to_datetime(args.dt_from), pd.to_datetime(args.dt_to)
    base = "https://content.guardianapis.com/search"
    page = 1
    out_rows = []

    progress(5, "Guardian: consultando…")
    while page <= 60:
        params = {
            "q": "Amazon",
            "from-date": fr.date().isoformat(),
            "to-date": to.date().isoformat(),
            "show-fields": "headline,trailText,byline,body",
            "page-size": 50,
            "page": page,
            "api-key": api_key,
            "order-by": "newest",
        }
        try:
            r = requests.get(base, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            progress(100, f"Error HTTP: {e}")
            print(json.dumps({"error": "http_error", "detail": str(e)}))
            sys.exit(2)

        resp = data.get("response") or {}
        results = resp.get("results") or []
        if not results:
            break

        for it in results:
            flds = it.get("fields") or {}
            title = flds.get("headline") or it.get("webTitle") or ""
            if not title_has_amazon(title):
                continue
            link = it.get("webUrl") or ""
            pub = it.get("webPublicationDate")
            dtv = pd.to_datetime(pub, errors="coerce", utc=True)
            if pd.isna(dtv):
                continue
            section = it.get("sectionName") or domain_from_link(link)
            summary = flds.get("trailText") or flds.get("body") or ""

            out_rows.append({
                "Fecha": dtv.tz_convert(TZ_ET).date(),
                "Titular": title.strip(),
                "Resumen": summarize(summary),
                "URL": link,
                "Seccion": section or "",
                "Desk": "",
                "Tipo": "Article",
                "Autor": (flds.get("byline") or ""),
                "Fuente": "Guardian",
            })

        pages = int(resp.get("pages") or 1)
        progress(min(95, int(5 + (page / max(1, pages)) * 80)), f"Guardian página {page}/{pages}")
        if page >= pages:
            break
        page += 1
        time.sleep(0.4)

    # Dedup básico
    if out_rows:
        df = pd.DataFrame(out_rows)
        df["__tnorm"] = df["Titular"].map(norm_title)
        df = df.sort_values("Fecha").drop_duplicates(subset=["URL"], keep="first")
        df = df.sort_values("Fecha").drop_duplicates(subset=["__tnorm"], keep="first")
        df = df.drop(columns=["__tnorm"])
    else:
        df = pd.DataFrame(columns=["Fecha","Titular","Resumen","URL","Seccion","Desk","Tipo","Autor","Fuente"])

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"guardian_{fr.date()}_{to.date()}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    progress(100, f"Guardian listo: {len(df)} filas")
    print(json.dumps({"csv": out_path, "count": int(df.shape[0]), "source": "Guardian",
                      "from": str(fr.date()), "to": str(to.date())}))

if __name__ == "__main__":
    main()
