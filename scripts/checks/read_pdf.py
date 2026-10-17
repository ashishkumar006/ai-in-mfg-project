from __future__ import annotations

import sys
from pathlib import Path

from pypdf import PdfReader

path = Path(sys.argv[1])
reader = PdfReader(str(path))
print(f"FILE   : {path.name}")
print(f"PAGES  : {len(reader.pages)}")
meta = reader.metadata or {}
if meta:
    print(f"TITLE  : {meta.get('/Title')}")
print("=" * 70)

for i, page in enumerate(reader.pages, 1):
    text = page.extract_text() or ""
    text = " ".join(text.split())
    print(f"\n----- page {i} ({len(text)} chars) -----")
    print(text if text else "[no extractable text — likely a scanned/image PDF]")