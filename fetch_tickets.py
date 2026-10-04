"""Fetch 票價 / 主辦單位 / 售票系統 from every arena.taipei program detail page -> events_tickets.csv.
Listing pages are read from external/arena_pages/*.html (re-fetch them if missing)."""
import re, html, csv, urllib.request, time
from concurrent.futures import ThreadPoolExecutor

BASE = "https://www.arena.taipei/"
items = []
for p in range(1, 26):
    h = open(f"external/arena_pages/{p}.html", encoding="utf-8", errors="ignore").read()
    for href, t in re.findall(r'<a\s+href="(News_Content\.aspx[^"]*)"[^>]*>(.*?)</a>', h, re.S):
        t = html.unescape(re.sub(r"\s+", " ", re.sub("<[^>]+>", "", t))).strip()
        if re.match(r"\d{4}/", t): items.append((html.unescape(href), t))

def get(href):
    for i in range(4):
        try:
            return urllib.request.urlopen(BASE + href, timeout=40).read().decode("utf-8", "ignore")
        except Exception:
            time.sleep(3 * (i + 1))
    return ""

def field(t, name, stops):
    name = r"\s*".join(name)
    m = re.search(name + r"\s*[：:](.*?)(?:" + "|".join(stops) + r"|資料更新|點閱數|$)", t)
    return m[1].strip() if m else ""

def work(it):
    href, title = it
    h = get(href)
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", h, flags=re.S)
    t = html.unescape(re.sub(r"\s+", " ", re.sub("<[^>]+>", " ", t)))
    t = t[t.find("活動日期"):] if "活動日期" in t else t
    org = field(t, "主辦單位", ["聯絡電話", "票價", "售票系統"])
    price = field(t, "票價", ["售票系統", "主辦單位", "聯絡電話"])
    sys_ = field(t, "售票系統", ["點閱數", "票價", "主辦單位"])
    return [title, BASE + href, org, price, sys_, "" if h else "FETCH_FAIL"]

with ThreadPoolExecutor(6) as ex:
    res = list(ex.map(work, items))
with open("events_tickets.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["title_raw", "url", "organizer", "price_raw", "ticket_system", "error"]); w.writerows(res)
print("items", len(res), "fail", sum(1 for r in res if r[5]), "no price", sum(1 for r in res if not r[3]))
