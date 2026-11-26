"""Pixel spacing (cm) per patient from the DICOM headers, for make_synthetic.py.
  python get_spacing.py data/Mayo2016 3 > data/spacing.json     (folder with <patient>/full_<mm>mm/*.IMA)
"""
import json
import os
import sys

import pydicom

root, mm = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "3"
out = {}
for p in sorted(os.listdir(root)):
    d = os.path.join(root, p, f"full_{mm}mm")
    if not os.path.isdir(d):
        continue
    f = sorted(os.listdir(d))[0]
    ds = pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True)
    sy, sx = (float(v) for v in ds.PixelSpacing)
    if abs(sy - sx) > 1e-6:
        print(f"WARNING {p}: anisotropic pixel spacing {sy} x {sx} mm", file=sys.stderr)
    out[p] = round(sx / 10.0, 6)                        # mm -> cm
print(json.dumps(out, indent=1))
