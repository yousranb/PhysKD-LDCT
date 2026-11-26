"""Mayo2016 slices stored as .npy files on the fixed scale x = (HU + 1024) / 4096.

Real pairs   : <patient>_<index>_input.npy (low dose) and <patient>_<index>_target.npy (full dose), 512 x 512.
Synthetic    : <patient>_<index>_<family>L<k>_input.npy and <patient>_<index>_cleanref.npy (make_synthetic.py).
No per-image normalization: every model sees the same values. Training uses random crops (same location in input
and target); validation uses a center crop of the same size.
"""
import glob
import os
import re

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

TRAIN_PATIENTS = ["L067", "L096", "L109", "L143", "L192", "L286", "L291", "L310"]
VAL_PATIENTS = ["L333"]
TEST_PATIENTS = ["L506"]


def list_pairs(npy_dir, patients):
    pairs = []
    for p in patients:
        # exact match: "L506" must not pick up "L506_10_3_input.npy" (10 % dose set)
        pat = re.compile(rf"^{re.escape(p)}_(\d+)_input\.npy$")
        inputs = [f for f in glob.glob(os.path.join(npy_dir, f"{p}_*_input.npy"))
                  if pat.match(os.path.basename(f))]
        if not inputs:
            raise FileNotFoundError(f"no {p}_<index>_input.npy in {npy_dir}")
        inputs.sort(key=lambda s: int(pat.match(os.path.basename(s)).group(1)))
        for inp in inputs:
            tgt = inp.replace("_input.npy", "_target.npy")
            if not os.path.exists(tgt):
                raise FileNotFoundError(f"missing target for {inp}")
            pairs.append((inp, tgt))
    return pairs


class NpyPairs(Dataset):
    def __init__(self, pairs, crop=None, random_crop=False):
        self.pairs, self.crop, self.random_crop = pairs, crop, random_crop

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        q = np.load(self.pairs[i][0]).astype(np.float32)
        f = np.load(self.pairs[i][1]).astype(np.float32)
        if self.crop:
            H, W = q.shape
            c = self.crop
            if self.random_crop:
                y, x = np.random.randint(0, H - c + 1), np.random.randint(0, W - c + 1)
            else:
                y, x = (H - c) // 2, (W - c) // 2
            q, f = q[y:y + c, x:x + c], f[y:y + c, x:x + c]
        return torch.from_numpy(q.copy())[None], torch.from_numpy(f.copy())[None]


def create_loaders_npy(npy_dir, train_patients, val_patients, crop, batch_size, num_workers=4):
    tr = NpyPairs(list_pairs(npy_dir, train_patients), crop=crop, random_crop=True)
    va = NpyPairs(list_pairs(npy_dir, val_patients), crop=crop, random_crop=False)
    print(f"[npy] train {train_patients}: {len(tr)} slices | val {val_patients}: {len(va)} slices | crop {crop}")
    return (DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
            DataLoader(va, batch_size=1, shuffle=False, num_workers=num_workers))


def list_syn_pairs(syn_dir, patients, families):
    """(degraded input, clean reference) pairs written by make_synthetic.py, for the given patients/families."""
    pat = re.compile(r"^([A-Za-z0-9]+)_(\d+)_(noise|blur|streak)(L\d)_input\.npy$")
    pairs = []
    for f in sorted(glob.glob(os.path.join(syn_dir, "*_input.npy"))):
        m = pat.match(os.path.basename(f))
        if not m or m.group(1) not in patients or m.group(3) not in families:
            continue
        t = os.path.join(syn_dir, f"{m.group(1)}_{m.group(2)}_cleanref.npy")
        if os.path.exists(t):
            pairs.append((f, t))
    if not pairs:
        raise FileNotFoundError(f"no synthetic pairs for {patients} / {families} in {syn_dir}")
    return pairs


def mix_synthetic(tr, a):
    """Real LDCT/FDCT training pairs + synthetic (degraded -> clean reference) pairs from make_synthetic.py.
    Each epoch keeps the same number of samples as without augmentation; a fraction --syn_ratio is synthetic.
    Validation and test stay on real data only."""
    pairs = list_syn_pairs(a.syn_dir, a.train_patients, a.syn_families)
    real, syn = tr.dataset, NpyPairs(pairs, crop=a.crop, random_crop=True)
    r = a.syn_ratio
    w = torch.cat([torch.full((len(real),), (1 - r) / len(real)), torch.full((len(syn),), r / len(syn))])
    sampler = WeightedRandomSampler(w.double(), num_samples=len(real), replacement=True)
    counts = {fam: sum(fam in os.path.basename(p[0]) for p in pairs) for fam in a.syn_families}
    print(f"[syn] {len(syn)} synthetic pairs {counts} from {a.syn_dir} | real {len(real)} | "
          f"{100 * r:.0f}% synthetic per epoch, {len(real)} samples/epoch")
    return DataLoader(ConcatDataset([real, syn]), batch_size=a.batch_size, sampler=sampler,
                      num_workers=a.workers, drop_last=True)
