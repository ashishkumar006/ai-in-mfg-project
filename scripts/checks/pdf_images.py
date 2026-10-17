from __future__ import annotations

import sys
from pathlib import Path

from pypdf import PdfReader

src = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

reader = PdfReader(str(src))
total = 0
for pageno, page in enumerate(reader.pages, 1):
    for idx, image in enumerate(page.images, 1):
        suffix = Path(image.name).suffix or ".png"
        target = out / f"page{pageno:02d}_{idx}{suffix}"
        target.write_bytes(image.data)
        total += 1
        print(f"{target.name}  {len(image.data) / 1024:.0f} KB")
print(f"\ntotal images: {total}")