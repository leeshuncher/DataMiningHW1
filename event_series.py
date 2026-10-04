"""Link event titles of the same artist / series (e.g. 周杰倫 tours, 金曲獎 each year) by shared distinctive name fragments.
Pairwise links only (no transitive closure) because fragments like 大同 or idol create false links that would chain."""
import re, collections, itertools
import pandas as pd

GENERIC = """演唱會 音樂會 巡迴 世界巡迴 世界 台北小巨蛋 小巨蛋 臺北小巨蛋 台北 臺北 北站 台北站 臺北站 台北場 臺北場 返場 安可場 安可 加場 追加場 追加
 特別版 週年 周年 年度 巨蛋 巡演 全球人壽 人壽 銀行 冠名贊助 冠名 贊助 台新銀行 live tour world taipei concert presents asia in 開唱 大大 大型 公演 場次
 演出 台灣 臺灣 學年度 學年 公開賽 頒獎典禮 典禮 決賽 聯賽 總決賽 盃 國際 流行音樂 中華民國 全國 氣密窗 大同氣密窗 限定 限定版 登場 初登場 首度 特別 紀念
 重返 全新 新世界 星球 夢想 愛情 時光 亞洲 中華 未來 活動 經典 表演 旗艦 最終 最終場 之夜 人生 我們 運動 運動會 音樂 電影
 金控 全球 藝術 之旅 大會 輝葉按 輝葉 按摩椅 旅程 如果 大哥大 台灣大哥大 非公開 活動 體育表演 冰上 音樂劇 富邦 同樂 慶典 嘉年華 之王 傳奇 精選 回憶 青春 """.split()
GENERIC = sorted(set(GENERIC), key=len, reverse=True)
LATIN_STOP = {"idol", "best", "power", "love", "dream", "dreams", "night", "show", "stage", "fans", "story", "music", "party",
              "special", "final", "classic", "festival", "zone", "star", "super", "game", "first", "again", "life", "hits", "moment",
              "limited", "edition", "taipei", "tour", "world", "live", "asia"}
MAX_DF = 14   # a fragment shared by more titles than this is treated as generic

def _clean(s):
    s = re.sub(r"\d+", "", s.lower())
    for g in GENERIC: s = s.replace(g, " ")
    return re.sub(r"[^一-鿿a-z]+", " ", s)

def _grams(s):
    g = set()
    for w in _clean(s).split():
        if re.fullmatch(r"[a-z]+", w):
            if len(w) >= 5 and w not in LATIN_STOP: g.add(w)
        else:
            for seg in re.findall(r"[一-鿿]+", w):
                for n in (2, 3, 4):
                    for i in range(len(seg) - n + 1): g.add(seg[i:i + n])
    return g

def title_links(titles):
    """titles: {raw_key: title}. Returns {raw_key: {raw_key: shared_fragment}} (symmetric)."""
    keys = list(titles); gr = {k: _grams(titles[k]) for k in keys}
    dfc = collections.Counter(x for k in keys for x in gr[k])
    out = collections.defaultdict(dict)
    for a, b in itertools.combinations(keys, 2):
        sh = [x for x in gr[a] & gr[b] if dfc[x] <= MAX_DF]
        if sh:
            best = max(sh, key=len); out[a][b] = best; out[b][a] = best
    return out

if __name__ == "__main__":
    o = pd.read_csv("events_official.csv", encoding="utf-8-sig").drop_duplicates("raw")
    tt = dict(zip(o.raw, o.title)); L = title_links(tt)
    pairs = sorted({(min(a, b), max(a, b), f) for a, d in L.items() for b, f in d.items()})
    print(len(tt), "titles;", sum(1 for k in tt if L.get(k)), "linked;", len(pairs), "pairs")
    for a, b, f in pairs: print(f"[{f}] {tt[a][:40]}  <->  {tt[b][:40]}")
