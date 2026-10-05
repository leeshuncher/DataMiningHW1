"""Download TDCS M04A (median travel time between adjacent gantries, 5-min) from
the Freeway Bureau archive and keep only Freeway 5 southbound rows.

Source : https://tisvcloud.freeway.gov.tw/history/TDCS/M04A/M04A_YYYYMMDD.tar.gz
Row    : time, from_gantry, to_gantry, vehicle_type, travel_time_sec, sample_count
Kept   : rows whose from AND to gantry both match 05F....S (Freeway 5 southbound
         main line), all vehicle types. The Nangang -> Toucheng endpoints are chosen
         later in build_travel_time.py, so nothing is dropped by gantry here.
Output : data/m04a/M04A_YYYYMMDD.csv (one file per day; existing files are skipped,
         so the script can be stopped and resumed).
Rate   : the site asks for >= 40 s between requests; we wait 41 s.

Usage  : python src/download_m04a.py 20240101 20241231
"""
import io
import re
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timedelta
from pathlib import Path

BASE = "https://tisvcloud.freeway.gov.tw/history/TDCS/M04A/M04A_{d}.tar.gz"
OUT = Path(__file__).resolve().parent.parent / "data" / "m04a"
GAP = 41
KEEP = re.compile(r"^05F\d{4}S$")


def fetch(day):
    # urllib/ssl rejects the site's certificate (missing Subject Key Identifier); curl accepts it.
    return subprocess.run(["curl", "-sS", "-f", "-m", "300", BASE.format(d=day)],
                          check=True, capture_output=True).stdout


def filter_day(blob):
    rows = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.name.endswith(".csv"):
                continue
            for line in tar.extractfile(m).read().decode("utf-8").splitlines():
                p = line.split(",")
                if len(p) == 6 and KEEP.match(p[1]) and KEEP.match(p[2]):
                    rows.append(line)
    rows.sort()
    return rows


def main(start, end):
    OUT.mkdir(parents=True, exist_ok=True)
    d, last = datetime.strptime(start, "%Y%m%d"), datetime.strptime(end, "%Y%m%d")
    while d <= last:
        day = d.strftime("%Y%m%d")
        d += timedelta(days=1)
        dest = OUT / f"M04A_{day}.csv"
        if dest.exists():
            continue
        for attempt in range(1, 4):
            try:
                rows = filter_day(fetch(day))
                break
            except Exception as e:  # network error, 404 for a missing day, bad archive
                print(f"{day} attempt {attempt}: {e}", flush=True)
                time.sleep(GAP)
        else:
            print(f"{day} FAILED", flush=True)
            continue
        dest.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"{day} {len(rows)} rows", flush=True)
        time.sleep(GAP)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
