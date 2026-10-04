"""Parse arena.taipei program listing, expand to single dates, and fill events/candidates_YYYY.csv."""
import re, html, datetime as dt, pandas as pd
from event_utils import kind

PAGES = "/tmp/claude-1007/-home-114-leeshuncher-DataMining-Hw1/10f7ac13-f153-471e-864b-f9ffac07c3db/scratchpad/pages"
rows = []
for p in range(1, 26):
    h = open(f"{PAGES}/{p}.html", encoding="utf-8", errors="ignore").read()
    for m in re.findall(r"<a[^>]*News_Content[^>]*>(.*?)</a>", h, re.S):
        t = html.unescape(re.sub(r"\s+", " ", re.sub("<[^>]+>", "", m))).strip()
        mm = re.match(r"(\d{4})/(.*?)\s*《(.*)》\s*$", t)
        if mm: rows.append((int(mm[1]), mm[2], mm[3], t))

def expand(year, spec):
    out, cur = [], (year, 1, 1)
    def md(s, base):  # "YYYY/M/D", "M/D" or "D" -> (y, m, d)
        a = [int(x) for x in s.split("/")]
        if len(a) == 3: return tuple(a)
        if len(a) == 2: return (base[0] + (a[0] < base[1]), a[0], a[1])
        return (base[0], base[1], a[0])
    for tok in re.split(r"[、,]", spec.replace(" ", "")):
        parts = re.split(r"[~\-]", tok)
        a = md(parts[0], cur); cur = a
        b = md(parts[1], a) if len(parts) > 1 and parts[1] else a
        d0, d1 = dt.date(*a), dt.date(*b)
        out += [d0 + dt.timedelta(i) for i in range((d1 - d0).days + 1)]
        cur = b
    return out

def clean(spec, year):
    """Handle notes: 取消 -> drop that date; 順延至X -> drop original, add X; leading text before first date is dropped."""
    spec = re.sub(r"^[^\d]*", "", spec) if not re.match(r"^[\d]", spec) else spec
    extra, toks = [], []
    for tok in re.split(r"[、,]", spec.replace(" ", "")):
        m = re.search(r"\((.*?)\)", tok); note = m[1] if m else ""
        tok = re.sub(r"\(.*?\)|[^\d/~\-]", "", tok) if not note.startswith("順延") else re.sub(r"\(.*?\)", "", tok)
        if "取消" in note: continue
        if note.startswith("順延"):
            extra.append(f"{year}/" + re.sub(r"[^\d/]", "", note)); continue
        if tok: toks.append(tok)
    return "、".join(toks + extra)

recs = []
for y, spec, title, raw in rows:
    if re.match(r"^\D", spec) and "取消" in spec: continue  # whole-day cancellation (e.g. typhoon)
    try:
        for d in expand(y, clean(spec, y)): recs.append((d, title, raw))
    except Exception as e:
        print("parse fail:", raw, e)
off = pd.DataFrame(recs, columns=["date", "title", "raw"]).drop_duplicates(["date", "title"])
off.to_csv("events_official.csv", index=False, encoding="utf-8-sig")
print("items", len(rows), "dates", len(off))

by = off.groupby("date")["title"].apply(lambda s: " / ".join(s))
tot = hit = 0
for y in range(2017, 2027):
    f = f"events/candidates_{y}.csv"
    c = pd.read_csv(f, encoding="utf-8-sig", dtype=str).fillna("")
    d = pd.to_datetime(c["日期"]).dt.date
    t = d.map(by).fillna("")
    c["活動名稱"] = t
    c["類型"] = t.map(lambda x: kind(x) if x else "")
    c["備註"] = ["" if x else "官網節目表無此日" for x in t]
    c.to_csv(f, index=False, encoding="utf-8-sig")
    tot += len(c); hit += (t != "").sum()
    print(y, len(c), "matched", (t != "").sum())
print("matched", hit, "of", tot)

# official event dates the detector did not flag
cand = set(pd.to_datetime(pd.concat([pd.read_csv(f"events/candidates_{y}.csv", encoding="utf-8-sig")["日期"] for y in range(2017, 2027)])).dt.date)
miss = off[(off.date >= dt.date(2017, 1, 1)) & (off.date <= dt.date(2026, 8, 31)) & ~off.date.isin(cand)]
miss.to_csv("events_official_not_flagged.csv", index=False, encoding="utf-8-sig")
print("official dates in range:", off[(off.date >= dt.date(2017,1,1)) & (off.date <= dt.date(2026,8,31))].date.nunique(), "not flagged:", miss.date.nunique())
