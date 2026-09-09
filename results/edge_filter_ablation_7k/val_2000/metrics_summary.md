# Thermal IR metric summary

All source encodings use a fixed mapping to [0,1] (uint8/255, uint16/65535); no per-image min-max normalization is used for intensities or edge weights. RGB is converted with BT.601 luma. LPIPS alone receives three replicated luma channels. E-MAE is normalized by GT edge-weight support. `unavailable` denotes skipped LPIPS or an absent/empty ROI.

| Experiment | Views | PSNR | SSIM | LPIPS | T-MAE | E-MAE | Gradient preservation | ROI-MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 34 | 29.56526259 | 0.94303549 | unavailable | 0.02825693 | 0.00672683 | 0.53757069 | unavailable |
| E2 | 34 | 29.64156454 | 0.94469732 | unavailable | 0.02694742 | 0.00675894 | 0.54179866 | unavailable |
| E2_no_filter | 34 | 29.61060719 | 0.94316956 | unavailable | 0.02779348 | 0.00669567 | 0.53955080 | unavailable |
