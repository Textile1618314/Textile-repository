#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CPSC_URL = "https://www.saferproducts.gov/RestWebServices/Recall?format=json"
CPSC_JSON = ROOT / "01_data" / "cpsc_recalls_all.json"
USER_AGENT = "Mozilla/5.0 (research; apparel-recalls reproduction)"
CHUNK = 1 << 20


def download(dest: Path = CPSC_JSON, url: str = CPSC_URL,
             timeout: int = 1800, quiet: bool = False) -> bool:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as out:
            total = r.headers.get("Content-Length")
            total = int(total) if total else None
            done = 0
            next_mark = 0
            while True:
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if not quiet and done >= next_mark:
                    if total:
                        print(f"  {done / 1e6:8.1f} MB / {total / 1e6:.1f} MB",
                              flush=True)
                    else:
                        print(f"  {done / 1e6:8.1f} MB", flush=True)
                    next_mark = done + 10 * CHUNK
        json.loads(tmp.read_text(encoding="utf-8", errors="replace"))
        tmp.replace(dest)
        if not quiet:
            print(f"saved {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return True
    except Exception as exc:
        if not quiet:
            print(f"download failed: {exc}", file=sys.stderr)
        if tmp.exists():
            tmp.unlink()
        return False


def main() -> int:
    if CPSC_JSON.exists() and CPSC_JSON.stat().st_size > 1_000_000:
        print(f"{CPSC_JSON} already present "
              f"({CPSC_JSON.stat().st_size / 1e6:.0f} MB); delete it to re-download")
        return 0
    print(f"downloading {CPSC_URL}")
    print(f"       into {CPSC_JSON}")
    return 0 if download() else 1


if __name__ == "__main__":
    sys.exit(main())
