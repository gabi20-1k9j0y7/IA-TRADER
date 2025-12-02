#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
thenewyorktimes.py — Descarga noticias de The New York Times sobre Amazon
Salida: CSV con columnas [Fecha, Titular, Resumen, URL, Seccion, Desk, Tipo, Autor, Fuente]
Progreso por stderr: líneas "%%<pct>|<mensaje>"
Salida final por stdout: JSON {"csv":..., "count":..., "source":"NYT","from":"YYYY-MM-DD","to":"YYYY-MM-DD"}
Comportamiento robusto: si no hay API key, si hay error HTTP o no hay resultados,
igualmente genera un CSV (posiblemente vacío) y devuelve rc=0.
"""
import os, sys, re, json, time, argparse
from datetime import timezone, timedelta
import pandas as pd
import requests

TZ_ET = timezone(timedelta(hours=-5))

def progress(pct: int, msg: str):
    sys.stderr.write(f"%%{pct}|{msg}\n"); sys.stderr.flush()

def get_api_key():
    try:
        import streamlit as st  # type: ignore
        secrets = dict(st.secrets) if hasattr(st, "secrets") else {}
    except Exception:
        secrets = {}
    return secrets.get("NYT_API_KEY") or os.getenv("NYT_API_KEY")

def title_has_amazon(title: str) -> bool:
    return isinstance(title, str) and bool(re.search(r"\bamazon\b", title, re.I))

def summarize(text: str, max_chars=420) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:max_chars]

def write_empty_csv(out_dir: str, fr, to):
    import os, pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"nyt_{fr.date()}_{to.date()}.csv")
    cols = ["Fecha","Titular","Resumen","URL","Seccion","Desk","Tipo","Autor","Fuente"]
    pd.DataFrame(columns=cols).to_csv(out_path, index=False, encoding="utf-8")
    return out_path

def main():
    ap = argparse.ArgumentParser("NYT fetcher (robusto)")
    ap.add_argument("--from", dest="dt_from", required=True)
    ap.add_argument("--to", dest="dt_to", required=True)
    ap.add_argument("--out-dir", dest="out_dir", required=True)
    args = ap.parse_args()

    fr, to = pd.to_datetime(args.dt_from), pd.to_datetime(args.dt_to)
    os.makedirs(args.out_dir, exist_ok=True)

    api_key = get_api_key()
    if not api_key:
        out_path = write_empty_csv(args.out_dir, fr, to)
        progress(100, "NYT: sin API key; CSV vacío generado")
        print(json.dumps({"csv": out_path, "count": 0, "source": "NYT",
                          "from": str(fr.date()), "to": str(to.date()), "note": "missing_api_key"}))
        sys.exit(0)

    url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
    page = 0
    rows = []
    progress(5, "NYT: consultando…")

    try:
        while page < 20:
            params = {
                "q": "Amazon",
                "begin_date": fr.strftime("%Y%m%d"),
                "end_date": to.strftime("%Y%m%d"),
                "api-key": api_key,
                "page": page,
                "sort": "newest"
            }
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            docs = (data.get("response") or {}).get("docs") or []
            if not docs:
                break

            for d in docs:
                title = ((d.get("headline") or {}).get("main")) or ""
                if not title_has_amazon(title):
                    continue
                link = (d.get("web_url") or "").strip()
                abstract = (d.get("abstract") or d.get("lead_paragraph") or "").strip()
                pub = d.get("pub_date")
                dtv = pd.to_datetime(pub, errors="coerce", utc=True)
                if pd.isna(dtv): 
                    continue
                section = d.get("section_name") or ""
                desk = d.get("news_desk") or ""
                typ = d.get("type_of_material") or "Article"
                by = ((d.get("byline") or {}).get("original") or "")

                rows.append({
                    "Fecha": dtv.tz_convert(TZ_ET).date(),
                    "Titular": title.strip(),
                    "Resumen": summarize(abstract),
                    "URL": link,
                    "Seccion": section,
                    "Desk": desk,
                    "Tipo": typ,
                    "Autor": by,
                    "Fuente": "NYT",
                })

            progress(min(95, 5 + page * 4), f"NYT página {page+1}")
            if len(docs) < 10:
                break
            page += 1
            time.sleep(0.4)
    except Exception as e:
        out_path = write_empty_csv(args.out_dir, fr, to)
        progress(100, f"NYT: error HTTP ({e}); CSV vacío generado")
        print(json.dumps({"csv": out_path, "count": 0, "source": "NYT",
                          "from": str(fr.date()), "to": str(to.date()), "note": "http_error"}))
        sys.exit(0)

    out_path = os.path.join(args.out_dir, f"nyt_{fr.date()}_{to.date()}.csv")
    if rows:
        df = pd.DataFrame(rows)
        df["__tnorm"] = df["Titular"].str.lower().str.replace(r"[^a-z0-9 ]+", "", regex=True)
        df = df.sort_values("Fecha").drop_duplicates(subset=["URL"], keep="first")
        df = df.sort_values("Fecha").drop_duplicates(subset=["__tnorm"], keep="first")
        df = df.drop(columns=["__tnorm"])
    else:
        df = pd.DataFrame(columns=["Fecha","Titular","Resumen","URL","Seccion","Desk","Tipo","Autor","Fuente"])
    df.to_csv(out_path, index=False, encoding="utf-8")
    progress(100, f"NYT listo: {len(df)} filas")
    print(json.dumps({"csv": out_path, "count": int(df.shape[0]), "source": "NYT",
                      "from": str(fr.date()), "to": str(to.date())}))
    sys.exit(0)

if __name__ == "__main__":
    main()
