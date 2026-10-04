"""Parse session times from events_detail.csv and fill 開演時間/預估散場 in events/candidates_YYYY.csv."""
import re, csv, datetime as dt, pandas as pd

def hm(s): h, m = s.split(":"); return int(h) * 60 + int(m)

sessions = []  # (date, raw_title, start, open, end)
for r in csv.DictReader(open("events_detail.csv", encoding="utf-8-sig")):
    f = r["time_field"]
    parts = re.split(r"(?=\d{4}/\d{1,2}/\d{1,2})", f)
    for p in parts:
        m = re.match(r"(\d{4})/(\d{1,2})/(\d{1,2})", p)
        if not m: continue
        try: d = dt.date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            try: d = dt.date(int(r["title_raw"][:4]), int(m[2]), int(m[3])); print("fixed typo date:", m[0], "->", d)
            except ValueError: print("bad date:", r["title_raw"][:40], m[0]); continue
        body = p[m.end():]
        body = re.sub(r"^\s*\(.\)\s*", "", body)          # weekday
        st = re.match(r"\s*(\d{1,2}:\d{2})", body)
        op = re.search(r"(\d{1,2}:\d{2})\s*開放入場", body)
        en = re.search(r"(\d{1,2}:\d{2})\s*活動結束", body)
        sessions.append((d, r["title_raw"], st[1] if st else "", op[1] if op else "", en[1] if en else ""))
S = pd.DataFrame(sessions, columns=["date", "title_raw", "start", "open", "end"]).drop_duplicates()
S.to_csv("events_sessions.csv", index=False, encoding="utf-8-sig")
print("sessions", len(S), "no start", (S.start == "").sum(), "no end", (S.end == "").sum())

def summarize(g):
    g = g[g.start != ""]
    starts = sorted(set(g.start), key=hm)
    ends = [(hm(e) + (1440 if hm(e) < hm(s) else 0), e) for s, e in zip(g.start, g.end) if e]
    last = max(ends)[1] if ends else ""
    return pd.Series({"開演時間": "/".join(starts), "預估散場": last, "場次": len(g)})

agg = S.groupby("date").apply(summarize, include_groups=False)
filled = tot = 0
for y in range(2017, 2027):
    f = f"events/candidates_{y}.csv"
    c = pd.read_csv(f, encoding="utf-8-sig", dtype=str).fillna("")
    d = pd.to_datetime(c["日期"]).dt.date
    a = agg.reindex(d.values)
    c["開演時間"] = a["開演時間"].fillna("").values
    c["預估散場"] = a["預估散場"].fillna("").values
    c["場次"] = a["場次"].fillna(0).astype(int).astype(str).replace("0", "").values
    cols = [x for x in c.columns if x != "場次"]
    cols.insert(cols.index("預估散場") + 1, "場次")
    c = c[cols]
    c.to_csv(f, index=False, encoding="utf-8-sig")
    named = (c["活動名稱"] != "").sum(); got = (c["預估散場"] != "").sum()
    print(y, "named", named, "with end time", got, "with start", (c["開演時間"] != "").sum()); tot += named; filled += got
print("named", tot, "with end", filled)
