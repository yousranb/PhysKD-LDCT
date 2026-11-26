Result tables of the paper (one row per method, thickness and dose; mean and SD; `SD_type` tells whether the SD is
between training seeds or between test slices; `Seeds` lists the runs). LPIPS and DISTS are on their native scale
here (the paper multiplies them by 100). Rows named `Full E1 ...` / `Full E6 ...` are the single-run full-size
exploratory experiments mentioned in the text of the paper; they are not reproduced by the scripts of this
repository. The WGAN-VGG and CTformer values at 25/10/5 % were exported rounded (2 decimals for PSNR/NQM,
4 for the other metrics). The DenoMamba NQM value is excluded from the paper (scale anomaly under verification).
