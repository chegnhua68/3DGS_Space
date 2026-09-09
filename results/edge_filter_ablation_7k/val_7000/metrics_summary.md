# Thermal IR metric summary

All source encodings use a fixed mapping to [0,1] (uint8/255, uint16/65535); no per-image min-max normalization is used for intensities or edge weights. RGB is converted with BT.601 luma. LPIPS alone receives three replicated luma channels. E-MAE is normalized by GT edge-weight support. `unavailable` denotes skipped LPIPS or an absent/empty ROI.

| Experiment | Views | PSNR | SSIM | LPIPS | T-MAE | E-MAE | Gradient preservation | ROI-MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 34 | 30.39051661 | 0.94341818 | unavailable | 0.02791338 | 0.00642283 | 0.53745191 | unavailable |
| E2 | 34 | 31.00450084 | 0.94794881 | unavailable | 0.02396479 | 0.00642512 | 0.52745980 | unavailable |
| E2_no_filter | 34 | 30.19295875 | 0.94501015 | unavailable | 0.02773011 | 0.00646913 | 0.52384254 | unavailable |
