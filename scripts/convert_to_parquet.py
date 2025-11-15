import sys
from pathlib import Path

import polars as pl

src = Path(sys.argv[1])
dst = Path("data") / (src.stem + ".parquet")
dst.parent.mkdir(parents=True, exist_ok=True)
df = pl.read_csv(src) if src.suffix.lower() == ".csv" else pl.read_parquet(src)
df.write_parquet(dst)
print(f"Saved -> {dst}")
