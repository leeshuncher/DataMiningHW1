"""Fetch each arena.taipei program detail page and save the 「活動日期/時間」 field."""
import re, html, csv, urllib.request, time
from concurrent.futures import ThreadPoolExecutor

SCRATCH = "/tmp/claude-1007/-home-114-leeshuncher-DataMining-Hw1/10f7ac13-f153-471e-864b-f9ffac07c3db/scratchpad"
BASE = "https://www.arena.taipei/"
items = []
for p in range(1, 26):
    h = open(f"{SCRATCH}/pages/{p}.html", encoding="utf-8", errors="ignore").read()
    for href, t in re.findall(r'<a\s+href="(News_Content\.aspx[^"]*)"[^>]*>(.*?)</a>', h, re.S):
        t = html.unescape(re.sub(r"\s+", " ", re.sub("<[^>]+>", "", t))).strip()
        if re.match(r"\d{4}/", t): items.append((html.unescape(href), t))
print("items", len(items))

def get(href):
    for i in range(4):
        try:
            return urllib.request.urlopen(BASE + href, timeout=40).read().decode("utf-8", "ignore")
        except Exception:
            time.sleep(3 * (i + 1))
    return ""

def work(it):
    href, title = it
    h = get(href)
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", h, flags=re.S)
    t = html.unescape(re.sub(r"\s+", " ", re.sub("<[^>]+>", " ", t)))
    m = re.search(r"活動日期\s*/\s*時間\s*[：:](.*?)(?:票價|主辦單位|聯絡電話|售票系統|點閱數)", t)
    return [title, BASE + href, m[1].strip() if m else "", "" if h else "FETCH_FAIL"]

with ThreadPoolExecutor(6) as ex:
    res = list(ex.map(work, items))
with open("events_detail.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["title_raw", "url", "time_field", "error"]); w.writerows(res)
print("no time field:", sum(1 for r in res if not r[2]), "fetch fail:", sum(1 for r in res if r[3]))
