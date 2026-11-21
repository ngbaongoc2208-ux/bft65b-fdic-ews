"""
Part 3 - helper: write a small Spark-friendly copy of the panel (11 columns).

Polars reads data_clean/panel.parquet, pyarrow writes data_clean/panel_spark.parquet (about 50 MB, snappy).
Reason: on this machine pyarrow/Spark could not parse the footer of the Polars-written file, so the copy
for Spark is written by a different Parquet writer.
Run   : python src/06_export_for_spark.py
Prints the size and MD5 so you can check that the file arrived intact in Google Colab.
"""
import hashlib
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

NEEDED = ["CERT", "qi", "ASSET", "EQ", "NCLNLS", "LNLSGR", "CHBAL", "SC", "LNLSNET", "DEP", "ROA"]
SRC, DST = Path("data_clean/panel.parquet"), Path("data_clean/panel_spark.parquet")


def footer_info(p: Path) -> str:
    with open(p, "rb") as fh:
        fh.seek(-8, 2)
        tail = fh.read(8)
    return f"size {p.stat().st_size:,} bytes, footer length {int.from_bytes(tail[:4], 'little'):,}, ends with {tail[4:]!r}"


print("Diagnostics")
for p in [SRC, Path("data_clean/bench/panel_x1.parquet")]:
    if not p.exists():
        continue
    try:
        pq.ParquetFile(p)
        status = "pyarrow can open it"
    except Exception as e:
        status = "pyarrow CANNOT open it: " + str(e).splitlines()[0][:80]
    print(f"  {p}: {footer_info(p)}; {status}")

df = pl.read_parquet(SRC, columns=NEEDED)          # Polars can read its own file
pq.write_table(df.to_arrow(), DST, compression="snappy")   # pyarrow writes the copy for Spark

h = hashlib.md5()
with open(DST, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
check = pq.read_table(DST)
print(f"\nWrote {DST}: {check.num_rows:,} rows x {check.num_columns} columns")
print(f"size {DST.stat().st_size / 1e6:.1f} MB   md5 {h.hexdigest()}")
print("Upload this file to Colab; the notebook prints the same size and md5 if the upload is intact.")