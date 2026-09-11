# Ablation summary — generated 2026-08-25 10:09

Root: `/media/Data03/Projects/Automated_Electrode_Coordinate_Extraction/code/Network/Cluster`  
Reports parsed: 56  
Log files scanned: 59  
Per-case source: 51 csv / 1 report

**Tags with an unrecognised network name** (add them to NETWORKS at the top of the script):

- `merged_ablation`
- `paper_attention_unet`

## 1. Coverage (held-out test Dice per fold)

| network | dataset | loss | f0 | f1 | f2 | f3 | f4 | non-CV | n |
|---|---|---|---|---|---|---|---|---|---|
| UNKNOWN | ? | dicefocal | **missing** | **missing** | **missing** | **missing** | **missing** | single:0.7185 | 0/5 |
| a_baseline | HGW | dicece | 0.6475 | 0.5927 | 0.6684 | 0.6479 | 0.6602 | -- | 5/5 |
| a_baseline | HGW | dicefocal | 0.4981 | 0.6461 | 0.6642 | 0.7061 | 0.6707 | -- | 5/5 |
| b_no_attention | HGW | dicece | 0.6644 | 0.5957 | 0.6803 | 0.6432 | 0.5616 | -- | 5/5 |
| b_no_attention | HGW | dicefocal | 0.6679 | 0.6606 | 0.6644 | 0.6528 | 0.6487 | -- | 5/5 |
| c_increased | HGW | dicefocal | -- | -- | **missing** | **missing** | **missing** | -- | 2/5 |
| c_reduced | HGW | dicece | 0.5619 | 0.6210 | 0.6492 | 0.6948 | 0.5374 | -- | 5/5 |
| c_reduced | HGW | dicefocal | 0.6238 | 0.6462 | 0.5539 | -- | 0.4574 | -- | 5/5 |
| d_increased | HGW | dicece | 0.6651 | 0.6312 | 0.6406 | 0.6844 | 0.5894 | -- | 5/5 |
| d_increased | HGW | dicefocal | 0.5157 | 0.5724 | 0.5304 | 0.6916 | 0.6629 | -- | 5/5 |
| proposed | HGW | dicece | 0.6640 | 0.6452 | 0.5498 | 0.6910 | 0.6335 | -- | 5/5 |
| proposed | HGW | dicefocal | 0.6435 | 0.6617 | 0.5624 | 0.6085 | 0.5925 | single:0.5650 | 5/5 |
| proposed | RU | dicefocal | **missing** | **missing** | **missing** | **missing** | **missing** | single:0.6111 | 0/5 |

## 2. Aggregates (CV folds only; mean of fold means)

_HD statistics exclude `inf` cases (empty predictions); their count is the last column._

| network | dataset | loss | n | mean Dice | SD | median case Dice | median HD | mean IoU | cases<0.3 | inf HD |
|---|---|---|---|---|---|---|---|---|---|---|
| a_baseline | HGW | dicece | 5 | 0.6433 | 0.0296 | 0.6883 | 6.40 | 0.4918 | 3/54 | 2 |
| a_baseline | HGW | dicefocal | 5 | 0.6370 | 0.0807 | 0.6943 | 23.37 | 0.4878 | 5/54 | 1 |
| b_no_attention | HGW | dicece | 5 | 0.6290 | 0.0493 | 0.6893 | 5.29 | 0.4788 | 3/54 | 0 |
| b_no_attention | HGW | dicefocal | 5 | 0.6589 | 0.0080 | 0.6780 | 6.08 | 0.4993 | 1/54 | 0 |
| c_reduced | HGW | dicece | 5 | 0.6129 | 0.0640 | 0.6698 | 7.00 | 0.4592 | 3/54 | 1 |
| c_reduced | HGW | dicefocal | 4 | 0.5703 | 0.0849 | 0.6412 | 21.83 | 0.4279 | 6/43 | 1 |
| d_increased | HGW | dicece | 5 | 0.6421 | 0.0361 | 0.6760 | 5.70 | 0.4887 | 3/54 | 0 |
| d_increased | HGW | dicefocal | 5 | 0.5946 | 0.0789 | 0.6904 | 7.79 | 0.4507 | 7/54 | 2 |
| proposed | HGW | dicece | 5 | 0.6367 | 0.0532 | 0.6787 | 5.74 | 0.4833 | 3/54 | 1 |
| proposed | HGW | dicefocal | 5 | 0.6137 | 0.0397 | 0.6970 | 8.37 | 0.4658 | 6/54 | 0 |

## 2b. Paired loss comparison (identical folds, vs dicefocal)

| network | dataset | alt loss | folds | dicefocal | alt | mean diff |
|---|---|---|---|---|---|---|
| a_baseline | HGW | dicece | 5 | 0.6370 | 0.6433 | +0.0063 |
| b_no_attention | HGW | dicece | 5 | 0.6589 | 0.6290 | -0.0298 |
| c_reduced | HGW | dicece | 4 | 0.5703 | 0.5924 | +0.0221 |
| d_increased | HGW | dicece | 5 | 0.5946 | 0.6421 | +0.0475 |
| proposed | HGW | dicece | 5 | 0.6137 | 0.6367 | +0.0230 |

_Paired means only. With n=5 folds, run a paired t-test or Wilcoxon before claiming a difference._

## 3. Per-run detail

| tag | net | ds | fold | loss | params | best val | @iter | test Dice | sd | test HD | test IoU | host |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `merged_ablation` | UNKNOWN | ? | single | dicefocal | **7,751,240** | -- | -- | -- | -- | -- | -- | -- |
| `paper_attention_unet` | UNKNOWN | ? | single | dicefocal | 23,625,986 | 0.6739 | 39500 | 0.7185 | 0.0480 | 12.92 | 0.5627 | vision-01 |
| `a_unet_baseline_HGW_cv5f0_seed1001_a_baseline_dicece` | a_baseline | HGW | 0 | dicece | 7,751,240 | 0.7151 | 32500 | 0.6475 | 0.2100 | inf | 0.5049 | vision-06 |
| `a_unet_baseline_HGW_cv5f1_seed1001_a_baseline_dicece` | a_baseline | HGW | 1 | dicece | 7,751,240 | 0.7092 | 25000 | 0.5927 | 0.1920 | inf | 0.4410 | vision-08 |
| `a_unet_baseline_HGW_cv5f2_seed1001_a_baseline_dicece` | a_baseline | HGW | 2 | dicece | 7,751,240 | 0.7183 | 41000 | 0.6684 | 0.0864 | 65.61 | 0.5080 | vision-08 |
| `a_unet_baseline_HGW_cv5f3_seed1001_a_baseline_dicece` | a_baseline | HGW | 3 | dicece | 7,751,240 | 0.7086 | 26000 | 0.6479 | 0.2084 | 58.30 | 0.5059 | vision-08 |
| `a_unet_baseline_HGW_cv5f4_seed1001_a_baseline_dicece` | a_baseline | HGW | 4 | dicece | 7,751,240 | 0.7107 | 50000 | 0.6602 | 0.0892 | 39.68 | 0.4992 | vision-08 |
| `a_unet_baseline_HGW_cv5f0_seed1001_a_baseline` | a_baseline | HGW | 0 | dicefocal | 7,751,240 | 0.7071 | 31000 | 0.4981 | 0.2763 | inf | 0.3732 | vision-01 |
| `a_unet_baseline_HGW_cv5f1_seed1001_a_baseline` | a_baseline | HGW | 1 | dicefocal | 7,751,240 | 0.7092 | 48000 | 0.6461 | 0.1830 | 48.12 | 0.4975 | vision-01 |
| `a_unet_baseline_HGW_cv5f2_seed1001_a_baseline` | a_baseline | HGW | 2 | dicefocal | 7,751,240 | 0.7117 | 25000 | 0.6642 | 0.0848 | 77.51 | 0.5032 | vision-01 |
| `a_unet_baseline_HGW_cv5f3_seed1001_a_baseline` | a_baseline | HGW | 3 | dicefocal | 7,751,240 | 0.7219 | 42500 | 0.7061 | 0.0829 | 68.45 | 0.5518 | vision-01 |
| `a_unet_baseline_HGW_cv5f4_seed1001_a_baseline` | a_baseline | HGW | 4 | dicefocal | 7,751,240 | 0.7137 | 49500 | 0.6707 | 0.0985 | 71.09 | 0.5131 | vision-01 |
| `b_proposed_no_attention_HGW_cv5f0_seed1001_b_no_attention_dicece` | b_no_attention | HGW | 0 | dicece | 7,912,874 | 0.7043 | 41500 | 0.6644 | 0.0975 | 51.10 | 0.5047 | vision-05 |
| `b_proposed_no_attention_HGW_cv5f1_seed1001_b_no_attention_dicece` | b_no_attention | HGW | 1 | dicece | 7,912,874 | 0.7032 | 38500 | 0.5957 | 0.2187 | 33.92 | 0.4522 | vision-05 |
| `b_proposed_no_attention_HGW_cv5f2_seed1001_b_no_attention_dicece` | b_no_attention | HGW | 2 | dicece | 7,912,874 | 0.7132 | 47000 | 0.6803 | 0.0690 | 31.39 | 0.5196 | vision-06 |
| `b_proposed_no_attention_HGW_cv5f3_seed1001_b_no_attention_dicece` | b_no_attention | HGW | 3 | dicece | 7,912,874 | 0.6995 | 48500 | 0.6432 | 0.2012 | 15.46 | 0.4980 | vision-06 |
| `b_proposed_no_attention_HGW_cv5f4_seed1001_b_no_attention_dicece` | b_no_attention | HGW | 4 | dicece | 7,912,874 | 0.7097 | 48000 | 0.5616 | 0.2255 | 35.60 | 0.4195 | vision-06 |
| `b_proposed_no_attention_HGW_cv5f0_seed1001_b_no_attention` | b_no_attention | HGW | 0 | dicefocal | 7,912,874 | 0.7124 | 41500 | 0.6679 | 0.1297 | 25.82 | 0.5137 | vision-01 |
| `b_proposed_no_attention_HGW_cv5f1_seed1001_b_no_attention` | b_no_attention | HGW | 1 | dicefocal | 7,912,874 | 0.7043 | 39500 | 0.6606 | 0.0670 | 46.40 | 0.4970 | vision-01 |
| `b_proposed_no_attention_HGW_cv5f2_seed1001_b_no_attention` | b_no_attention | HGW | 2 | dicefocal | 7,912,874 | 0.7060 | 33500 | 0.6644 | 0.0743 | 28.17 | 0.5021 | vision-01 |
| `b_proposed_no_attention_HGW_cv5f3_seed1001_b_no_attention` | b_no_attention | HGW | 3 | dicefocal | 7,912,874 | 0.7069 | 33500 | 0.6528 | 0.1328 | 18.61 | 0.4964 | vision-05 |
| `b_proposed_no_attention_HGW_cv5f4_seed1001_b_no_attention` | b_no_attention | HGW | 4 | dicefocal | 7,912,874 | 0.7068 | 32500 | 0.6487 | 0.1001 | 33.93 | 0.4873 | vision-05 |
| `paper_attention_unet_HGW_cv5f0_seed1001_c_increased` | c_increased | HGW | 0 | dicefocal | 53,150,586 | 0.0284 | 4000 | -- | -- | -- | -- | vision-01 |
| `paper_attention_unet_HGW_cv5f1_seed1001_c_increased` | c_increased | HGW | 1 | dicefocal | 53,150,586 | 0.0065 | 2500 | -- | -- | -- | -- | vision-01 |
| `attention_unet_reduced_HGW_cv5f0_seed1001_c_reduced_dicece` | c_reduced | HGW | 0 | dicece | 5,909,130 | 0.6891 | 40500 | 0.5619 | 0.2156 | inf | 0.4176 | vision-01 |
| `attention_unet_reduced_HGW_cv5f1_seed1001_c_reduced_dicece` | c_reduced | HGW | 1 | dicece | 5,909,130 | 0.6426 | 47500 | 0.6210 | 0.2057 | 21.91 | 0.4749 | vision-07 |
| `attention_unet_reduced_HGW_cv5f2_seed1001_c_reduced_dicece` | c_reduced | HGW | 2 | dicece | 5,909,130 | 0.6963 | 49500 | 0.6492 | 0.0925 | 43.55 | 0.4874 | vision-07 |
| `attention_unet_reduced_HGW_cv5f3_seed1001_c_reduced_dicece` | c_reduced | HGW | 3 | dicece | 5,909,130 | 0.7019 | 49500 | 0.6948 | 0.0423 | 42.81 | 0.5339 | vision-07 |
| `attention_unet_reduced_HGW_cv5f4_seed1001_c_reduced_dicece` | c_reduced | HGW | 4 | dicece | 5,909,130 | 0.6746 | 31000 | 0.5374 | 0.1535 | 73.41 | 0.3822 | vision-07 |
| `attention_unet_reduced_HGW_cv5f0_seed1001_c_reduced` | c_reduced | HGW | 0 | dicefocal | 5,909,130 | 0.6676 | 49000 | 0.6238 | 0.2124 | 33.92 | 0.4802 | vision-05 |
| `attention_unet_reduced_HGW_cv5f1_seed1001_c_reduced` | c_reduced | HGW | 1 | dicefocal | 5,909,130 | 0.6367 | 49500 | 0.6462 | 0.0774 | 19.37 | 0.4820 | vision-05 |
| `attention_unet_reduced_HGW_cv5f2_seed1001_c_reduced` | c_reduced | HGW | 2 | dicefocal | 5,909,130 | 0.6790 | 49500 | 0.5539 | 0.2768 | 73.24 | 0.4255 | vision-08 |
| `attention_unet_reduced_HGW_cv5f3_seed1001_c_reduced` | c_reduced | HGW | 3 | dicefocal | 5,909,130 | 0.0013 | 1000 | -- | -- | -- | -- | vision-08 |
| `attention_unet_reduced_HGW_cv5f4_seed1001_c_reduced` | c_reduced | HGW | 4 | dicefocal | 5,909,130 | 0.6725 | 44000 | 0.4574 | 0.2307 | inf | 0.3241 | vision-08 |
| `paper_attention_unet_HGW_cv5f0_seed1001_d_increased_dicece` | d_increased | HGW | 0 | dicece | 53,150,586 | 0.7007 | 48500 | 0.6651 | 0.1603 | 38.79 | 0.5162 | vision-03 |
| `paper_attention_unet_HGW_cv5f1_seed1001_d_increased_dicece` | d_increased | HGW | 1 | dicece | 53,150,586 | 0.7020 | 49000 | 0.6312 | 0.1165 | 25.31 | 0.4705 | vision-03 |
| `paper_attention_unet_HGW_cv5f2_seed1001_d_increased_dicece` | d_increased | HGW | 2 | dicece | 53,150,586 | 0.7000 | 47500 | 0.6406 | 0.1617 | 20.55 | 0.4878 | vision-03 |
| `paper_attention_unet_HGW_cv5f3_seed1001_d_increased_dicece` | d_increased | HGW | 3 | dicece | 53,150,586 | 0.6940 | 31500 | 0.6844 | 0.0689 | 35.51 | 0.5241 | vision-03 |
| `paper_attention_unet_HGW_cv5f4_seed1001_d_increased_dicece` | d_increased | HGW | 4 | dicece | 53,150,586 | 0.7158 | 49000 | 0.5894 | 0.2173 | 34.19 | 0.4451 | vision-01 |
| `paper_attention_unet_HGW_cv5f0_seed1001_d_increased` | d_increased | HGW | 0 | dicefocal | 53,150,586 | 0.6957 | 49000 | 0.5157 | 0.2849 | inf | 0.3919 | vision-08 |
| `paper_attention_unet_HGW_cv5f1_seed1001_d_increased` | d_increased | HGW | 1 | dicefocal | 53,150,586 | 0.6984 | 37500 | 0.5724 | 0.2029 | inf | 0.4241 | vision-03 |
| `paper_attention_unet_HGW_cv5f2_seed1001_d_increased` | d_increased | HGW | 2 | dicefocal | 53,150,586 | 0.6973 | 49000 | 0.5304 | 0.2821 | 55.50 | 0.4051 | vision-03 |
| `paper_attention_unet_HGW_cv5f3_seed1001_d_increased` | d_increased | HGW | 3 | dicefocal | 53,150,586 | 0.7046 | 48000 | 0.6916 | 0.0505 | 42.69 | 0.5307 | vision-03 |
| `paper_attention_unet_HGW_cv5f4_seed1001_d_increased` | d_increased | HGW | 4 | dicefocal | 53,150,586 | 0.7002 | 49500 | 0.6629 | 0.0875 | 38.49 | 0.5019 | vision-03 |
| `paper_attention_unet_HGW_cv5f0_seed1001_proposed_dicece` | proposed | HGW | 0 | dicece | 23,625,986 | 0.7038 | 49500 | 0.6640 | 0.0899 | 35.77 | 0.5035 | vision-01 |
| `paper_attention_unet_HGW_cv5f1_seed1001_proposed_dicece` | proposed | HGW | 1 | dicece | 23,625,986 | 0.6939 | 47000 | 0.6452 | 0.1511 | 39.04 | 0.4916 | vision-01 |
| `paper_attention_unet_HGW_cv5f2_seed1001_proposed_dicece` | proposed | HGW | 2 | dicece | 23,625,986 | 0.7172 | 49000 | 0.5498 | 0.2468 | inf | 0.4137 | vision-01 |
| `paper_attention_unet_HGW_cv5f3_seed1001_proposed_dicece` | proposed | HGW | 3 | dicece | 23,625,986 | 0.7111 | 50000 | 0.6910 | 0.0694 | 26.89 | 0.5322 | vision-05 |
| `paper_attention_unet_HGW_cv5f4_seed1001_proposed_dicece` | proposed | HGW | 4 | dicece | 23,625,986 | 0.7023 | 42500 | 0.6335 | 0.1299 | 36.02 | 0.4757 | vision-05 |
| `paper_attention_unet_HGW_cv5f0_seed1001_proposed` | proposed | HGW | 0 | dicefocal | 23,625,986 | 0.6935 | 48000 | 0.6435 | 0.1744 | 32.06 | 0.4949 | vision-07 |
| `paper_attention_unet_HGW_cv5f1_seed1001_proposed` | proposed | HGW | 1 | dicefocal | 23,625,986 | 0.6286 | 49000 | 0.6617 | 0.0634 | 41.74 | 0.4978 | vision-07 |
| `paper_attention_unet_HGW_cv5f2_seed1001_proposed` | proposed | HGW | 2 | dicefocal | 23,625,986 | 0.6691 | 40500 | 0.5624 | 0.2483 | 24.09 | 0.4258 | vision-07 |
| `paper_attention_unet_HGW_cv5f3_seed1001_proposed` | proposed | HGW | 3 | dicefocal | 23,625,986 | 0.6848 | 49500 | 0.6085 | 0.2072 | 65.52 | 0.4641 | vision-07 |
| `paper_attention_unet_HGW_cv5f4_seed1001_proposed` | proposed | HGW | 4 | dicefocal | 23,625,986 | 0.6538 | 40500 | 0.5925 | 0.2113 | 53.97 | 0.4462 | vision-06 |
| `paper_attention_unet_HGW_seed1001_proposed` | proposed | HGW | single | dicefocal | 23,625,986 | 0.6183 | 37000 | 0.5650 | 0.2186 | 40.48 | 0.4190 | vision-06 |
| `paper_attention_unet_RU_seed1001_proposed` | proposed | RU | single | dicefocal | 23,625,986 | 0.6116 | 33000 | 0.6111 | 0.2263 | 43.45 | 0.4698 | vision-06 |

## 4. Configuration check

_Any field with more than one distinct value across runs of the SAME network is a red flag._


**params**

- `23,625,986` — UNKNOWN, proposed
- `53,150,586` — c_increased, d_increased
- `**7,751,240**` — UNKNOWN
- `5,909,130` — c_reduced
- `7,751,240` — a_baseline
- `7,912,874` — b_no_attention

**channels**

- `(32, 64, 128, 256, 512)` — UNKNOWN, proposed
- `(48, 96, 192, 384, 768)` — c_increased, d_increased
- `(16, 32, 64, 128, 256)` — c_reduced
- `(32, 64, 128, 256, 512)  # matches proposed model` — b_no_attention
- `(64, 128, 256, 512)` — a_baseline
- `None` — UNKNOWN

**architecture**

- `AttentionUnet (MONAI)` — UNKNOWN, c_increased, d_increased, proposed
- `AttentionUnet (MONAI) — reduced channel depth` — c_reduced
- `MONAI UNet (proposed channel depths, NO attention gates, NO residual units)` — b_no_attention
- `MONAI UNet — classic Ronneberger-style baseline (NO attention)` — a_baseline
- `None` — UNKNOWN

**loss_label**

- `DiceFocalLoss (lambda_dice=0.3, lambda_focal=0.7, gamma=2.5, include_background=False)` — UNKNOWN, a_baseline, b_no_attention, c_increased, c_reduced, d_increased, proposed
- `DiceCELoss (lambda_dice=0.5, lambda_ce=0.5, weight=[1,10], include_background=False)` — a_baseline, b_no_attention, c_reduced, d_increased, proposed
- `None` — UNKNOWN

**lr**

- `1e-3 (Adam default; --lr ignored)` — UNKNOWN, a_baseline, b_no_attention, c_increased, c_reduced, d_increased, proposed
- `None` — UNKNOWN

**optimizer**

- `Adam (paper bc3d: lr=0.0005, wd=1e-4)` — a_baseline, b_no_attention, c_increased, c_reduced, d_increased, proposed
- `Adam (paper bc3d: lr=0.001, wd=1e-4)` — UNKNOWN
- `None` — UNKNOWN

**max_iter**

- `50000` — UNKNOWN, a_baseline, b_no_attention, c_increased, c_reduced, d_increased, proposed
- `None` — UNKNOWN

**eval_num**

- `500` — UNKNOWN, a_baseline, b_no_attention, c_increased, c_reduced, d_increased, proposed
- `None` — UNKNOWN

**Parameter count per network** (should be exactly one value each)

- UNKNOWN: **7,751,240**, 23,625,986  <-- INCONSISTENT
- a_baseline: 7,751,240
- b_no_attention: 7,912,874
- c_increased: 53,150,586
- c_reduced: 5,909,130
- d_increased: 53,150,586
- proposed: 23,625,986

## 5. Job status from logs

Completed: 51 / 59

| log | status | approx iterations |
|---|---|---|
| `b_no_attention_RU_FINAL_seed1001_7851606.out` | incomplete / running | 50500 |
| `c_reduced_HGW_cv5_seed1001_7736894_3.out` | SANITY ABORT | 21500 |
| `d_increased_HGW_smoketest_f0_seed9001_7736575.out` | incomplete / running | 3500 |
| `proposed_HGW_dicece_f2_seed1001_7811470.out` | incomplete / running | 0 |
| `proposed_HGW_dicece_f2_seed1001_7845745.out` | incomplete / running | 12000 |
| `proposed_HGW_dicece_f4_seed1001_7811471.out` | incomplete / running | 0 |
| `proposed_HGW_dicece_f4_seed1001_7845746.out` | incomplete / running | 12000 |
| `proposed_RU_FINAL_seed1001_7851604.out` | incomplete / running | 50500 |

## 6. Per-case test metrics

| network | loss | fold | idx | subject | file | Dice | HD | IoU |
|---|---|---|---|---|---|---|---|---|
| UNKNOWN | dicefocal | single | 0 | -- | -- | 0.7594 | 27.87 | 0.6121 |
| UNKNOWN | dicefocal | single | 1 | -- | -- | 0.7724 | 3.16 | 0.6291 |
| UNKNOWN | dicefocal | single | 2 | -- | -- | 0.7165 | 3.61 | 0.5582 |
| UNKNOWN | dicefocal | single | 3 | -- | -- | 0.7367 | 2.83 | 0.5832 |
| UNKNOWN | dicefocal | single | 4 | -- | -- | 0.7552 | 50.65 | 0.6067 |
| UNKNOWN | dicefocal | single | 5 | -- | -- | 0.7012 | 4.58 | 0.5399 |
| UNKNOWN | dicefocal | single | 6 | -- | -- | 0.6942 | 6.08 | 0.5316 |
| UNKNOWN | dicefocal | single | 7 | -- | -- | 0.6123 | 4.58 | 0.4413 |
| a_baseline | dicece | 0 | 0 | sub-011 | rsub-011_ses-3_acq-petra_run-01_PDw.nii | 0.7560 | 5.10 | 0.6077 |
| a_baseline | dicece | 0 | 1 | sub-012 | rsub-012_ses-4_acq-petra_run-02_PDw.nii | 0.7183 | 4.12 | 0.5605 |
| a_baseline | dicece | 0 | 2 | sub-014 | rsub-014_ses-4_acq-petra_run-02_PDw.nii | 0.0000 | inf | 0.0000 |
| a_baseline | dicece | 0 | 3 | sub-022 | rsub-022_ses-1_acq-petra_run-02_PDw.nii | 0.7990 | 5.00 | 0.6653 |
| a_baseline | dicece | 0 | 4 | sub-022 | rsub-022_ses-2_acq-petra_run-02_PDw.nii | 0.6707 | 107.31 | 0.5045 |
| a_baseline | dicece | 0 | 5 | sub-022 | rsub-022_ses-3_acq-petra_run-01_PDw.nii | 0.6775 | 205.03 | 0.5123 |
| a_baseline | dicece | 0 | 6 | sub-022 | rsub-022_ses-4_acq-petra_run-01_PDw.nii | 0.6329 | 224.81 | 0.4629 |
| a_baseline | dicece | 0 | 7 | sub-030 | rsub-030_ses-2_acq-petra_run-01_PDw.nii | 0.6773 | 207.43 | 0.5121 |
| a_baseline | dicece | 0 | 8 | sub-030 | rsub-030_ses-3_acq-petra_run-02_PDw.nii | 0.7205 | 4.00 | 0.5632 |
| a_baseline | dicece | 0 | 9 | sub-040 | rsub-040_ses-2_acq-petra_run-01_PDw.nii | 0.6987 | 3.00 | 0.5369 |
| a_baseline | dicece | 0 | 10 | sub-044 | rsub-044_ses-2_acq-petra_run-02_PDw.nii | 0.7719 | 203.03 | 0.6285 |
| a_baseline | dicece | 1 | 0 | sub-001 | rsub-001_ses-3_acq-petra_run-01_PDw.nii | 0.0000 | inf | 0.0000 |
| a_baseline | dicece | 1 | 1 | sub-001 | rsub-001_ses-4_acq-petra_run-02_PDw.nii | 0.6572 | 4.12 | 0.4894 |
| a_baseline | dicece | 1 | 2 | sub-002 | rsub-002_ses-1_acq-petra_run-01_PDw.nii | 0.6562 | 3.32 | 0.4883 |
| a_baseline | dicece | 1 | 3 | sub-002 | rsub-002_ses-1_acq-petra_run-02_PDw.nii | 0.6506 | 3.32 | 0.4821 |
| a_baseline | dicece | 1 | 4 | sub-002 | rsub-002_ses-2_acq-petra_run-01_PDw.nii | 0.6469 | 94.78 | 0.4780 |
| a_baseline | dicece | 1 | 5 | sub-006 | rsub-006_ses-2_acq-petra_run-01_PDw.nii | 0.6595 | 5.00 | 0.4920 |
| a_baseline | dicece | 1 | 6 | sub-015 | rsub-015_ses-1_acq-petra_run-02_PDw.nii | 0.7337 | 7.07 | 0.5794 |
| a_baseline | dicece | 1 | 7 | sub-021 | rsub-021_ses-3_acq-petra_run-01_PDw.nii | 0.5511 | 8.25 | 0.3804 |
| a_baseline | dicece | 1 | 8 | sub-024 | rsub-024_ses-1_acq-petra_run-01_PDw.nii | 0.6580 | 79.71 | 0.4903 |
| a_baseline | dicece | 1 | 9 | sub-038 | rsub-038_ses-4_acq-petra_run-01_PDw.nii | 0.6204 | 12.25 | 0.4497 |
| a_baseline | dicece | 1 | 10 | sub-052 | rsub-052_ses-4_acq-petra_run-01_PDw.nii | 0.6857 | 6.40 | 0.5218 |
| a_baseline | dicece | 2 | 0 | sub-009 | rsub-009_ses-1_acq-petra_run-01_PDw.nii | 0.7105 | 5.10 | 0.5509 |
| a_baseline | dicece | 2 | 1 | sub-018 | rsub-018_ses-4_acq-petra_run-01_PDw.nii | 0.5187 | 184.31 | 0.3502 |
| a_baseline | dicece | 2 | 2 | sub-023 | rsub-023_ses-2_acq-petra_run-02_PDw.nii | 0.7155 | 53.24 | 0.5571 |
| a_baseline | dicece | 2 | 3 | sub-029 | rsub-029_ses-1_acq-petra_run-01_PDw.nii | 0.6922 | 4.69 | 0.5293 |
| a_baseline | dicece | 2 | 4 | sub-029 | rsub-029_ses-3_acq-petra_run-02_PDw.nii | 0.7660 | 85.45 | 0.6208 |
| a_baseline | dicece | 2 | 5 | sub-036 | rsub-036_ses-4_acq-petra_run-01_PDw.nii | 0.7493 | 158.27 | 0.5991 |
| a_baseline | dicece | 2 | 6 | sub-043 | rsub-043_ses-1_acq-petra_run-02_PDw.nii | 0.7363 | 3.00 | 0.5826 |
| a_baseline | dicece | 2 | 7 | sub-054 | rsub-054_ses-2_acq-petra_run-01_PDw.nii | 0.5578 | 14.32 | 0.3868 |
| a_baseline | dicece | 2 | 8 | sub-054 | rsub-054_ses-2_acq-petra_run-02_PDw.nii | 0.5309 | 12.88 | 0.3614 |
| a_baseline | dicece | 2 | 9 | sub-054 | rsub-054_ses-3_acq-petra_run-02_PDw.nii | 0.6508 | 5.92 | 0.4824 |
| a_baseline | dicece | 2 | 10 | sub-073 | rsub-073_ses-2_acq-petra_run-01_PDw.nii | 0.7242 | 194.53 | 0.5677 |
| a_baseline | dicece | 3 | 0 | sub-004 | rsub-004_ses-1_acq-petra_run-02_PDw.nii | 0.5938 | 186.71 | 0.4222 |
| a_baseline | dicece | 3 | 1 | sub-016 | rsub-016_ses-2_acq-petra_run-01_PDw.nii | 0.7312 | 4.12 | 0.5763 |
| a_baseline | dicece | 3 | 2 | sub-016 | rsub-016_ses-2_acq-petra_run-02_PDw.nii | 0.0291 | 143.62 | 0.0148 |
| a_baseline | dicece | 3 | 3 | sub-016 | rsub-016_ses-3_acq-petra_run-01_PDw.nii | 0.7665 | 3.00 | 0.6214 |
| a_baseline | dicece | 3 | 4 | sub-017 | rsub-017_ses-3_acq-petra_run-01_PDw.nii | 0.7486 | 6.40 | 0.5983 |
| a_baseline | dicece | 3 | 5 | sub-017 | rsub-017_ses-4_acq-petra_run-01_PDw.nii | 0.7766 | 30.41 | 0.6347 |
| a_baseline | dicece | 3 | 6 | sub-032 | rsub-032_ses-3_acq-petra_run-01_PDw.nii | 0.5378 | 180.39 | 0.3678 |
| a_baseline | dicece | 3 | 7 | sub-034 | rsub-034_ses-4_acq-petra_run-02_PDw.nii | 0.7503 | 75.51 | 0.6003 |
| a_baseline | dicece | 3 | 8 | sub-037 | rsub-037_ses-4_acq-petra_run-01_PDw.nii | 0.7316 | 3.74 | 0.5768 |
| a_baseline | dicece | 3 | 9 | sub-058 | rsub-058_ses-3_acq-petra_run-01_PDw.nii | 0.7092 | 3.74 | 0.5494 |
| a_baseline | dicece | 3 | 10 | sub-062 | rsub-062_ses-4_acq-petra_run-01_PDw.nii | 0.7523 | 3.61 | 0.6030 |
| a_baseline | dicece | 4 | 0 | sub-003 | rsub-003_ses-3_acq-petra_run-02_PDw.nii | 0.7300 | 23.85 | 0.5748 |
| a_baseline | dicece | 4 | 1 | sub-005 | rsub-005_ses-4_acq-petra_run-02_PDw.nii | 0.5945 | 135.27 | 0.4230 |
| a_baseline | dicece | 4 | 2 | sub-007 | rsub-007_ses-3_acq-petra_run-02_PDw.nii | 0.6909 | 5.74 | 0.5278 |
| a_baseline | dicece | 4 | 3 | sub-007 | rsub-007_ses-4_acq-petra_run-01_PDw.nii | 0.5028 | 4.12 | 0.3358 |
| a_baseline | dicece | 4 | 4 | sub-010 | rsub-010_ses-3_acq-petra_run-01_PDw.nii | 0.6397 | 3.32 | 0.4703 |
| a_baseline | dicece | 4 | 5 | sub-010 | rsub-010_ses-4_acq-petra_run-01_PDw.nii | 0.6621 | 4.12 | 0.4949 |
| a_baseline | dicece | 4 | 6 | sub-013 | rsub-013_ses-1_acq-petra_run-01_PDw.nii | 0.7835 | 3.16 | 0.6441 |
| a_baseline | dicece | 4 | 7 | sub-013 | rsub-013_ses-2_acq-petra_run-02_PDw.nii | 0.7385 | 6.16 | 0.5854 |
| a_baseline | dicece | 4 | 8 | sub-020 | rsub-020_ses-2_acq-petra_run-02_PDw.nii | 0.5272 | 206.16 | 0.3580 |
| a_baseline | dicece | 4 | 9 | sub-066 | rsub-066_ses-1_acq-petra_run-02_PDw.nii | 0.7330 | 4.90 | 0.5785 |
| a_baseline | dicefocal | 0 | 0 | sub-011 | rsub-011_ses-3_acq-petra_run-01_PDw.nii | 0.7105 | 62.31 | 0.5510 |
| a_baseline | dicefocal | 0 | 1 | sub-012 | rsub-012_ses-4_acq-petra_run-02_PDw.nii | 0.7281 | 128.55 | 0.5725 |
| a_baseline | dicefocal | 0 | 2 | sub-014 | rsub-014_ses-4_acq-petra_run-02_PDw.nii | 0.0000 | inf | 0.0000 |
| a_baseline | dicefocal | 0 | 3 | sub-022 | rsub-022_ses-1_acq-petra_run-02_PDw.nii | 0.0974 | 234.04 | 0.0512 |
| a_baseline | dicefocal | 0 | 4 | sub-022 | rsub-022_ses-2_acq-petra_run-02_PDw.nii | 0.6076 | 171.17 | 0.4364 |
| a_baseline | dicefocal | 0 | 5 | sub-022 | rsub-022_ses-3_acq-petra_run-01_PDw.nii | 0.6615 | 5.39 | 0.4942 |
| a_baseline | dicefocal | 0 | 6 | sub-022 | rsub-022_ses-4_acq-petra_run-01_PDw.nii | 0.2837 | 125.59 | 0.1653 |
| a_baseline | dicefocal | 0 | 7 | sub-030 | rsub-030_ses-2_acq-petra_run-01_PDw.nii | 0.7205 | 151.03 | 0.5631 |
| a_baseline | dicefocal | 0 | 8 | sub-030 | rsub-030_ses-3_acq-petra_run-02_PDw.nii | 0.7287 | 3.00 | 0.5731 |
| a_baseline | dicefocal | 0 | 9 | sub-040 | rsub-040_ses-2_acq-petra_run-01_PDw.nii | 0.7390 | 69.76 | 0.5861 |
| a_baseline | dicefocal | 0 | 10 | sub-044 | rsub-044_ses-2_acq-petra_run-02_PDw.nii | 0.2018 | 197.23 | 0.1123 |
| a_baseline | dicefocal | 1 | 0 | sub-001 | rsub-001_ses-3_acq-petra_run-01_PDw.nii | 0.7619 | 129.83 | 0.6154 |
| a_baseline | dicefocal | 1 | 1 | sub-001 | rsub-001_ses-4_acq-petra_run-02_PDw.nii | 0.6154 | 146.84 | 0.4445 |
| a_baseline | dicefocal | 1 | 2 | sub-002 | rsub-002_ses-1_acq-petra_run-01_PDw.nii | 0.0801 | 200.18 | 0.0417 |
| a_baseline | dicefocal | 1 | 3 | sub-002 | rsub-002_ses-1_acq-petra_run-02_PDw.nii | 0.6902 | 3.61 | 0.5270 |
| a_baseline | dicefocal | 1 | 4 | sub-002 | rsub-002_ses-2_acq-petra_run-01_PDw.nii | 0.6786 | 4.12 | 0.5135 |
| a_baseline | dicefocal | 1 | 5 | sub-006 | rsub-006_ses-2_acq-petra_run-01_PDw.nii | 0.7031 | 15.84 | 0.5422 |
| a_baseline | dicefocal | 1 | 6 | sub-015 | rsub-015_ses-1_acq-petra_run-02_PDw.nii | 0.7256 | 6.00 | 0.5694 |
| a_baseline | dicefocal | 1 | 7 | sub-021 | rsub-021_ses-3_acq-petra_run-01_PDw.nii | 0.7348 | 5.66 | 0.5808 |
| a_baseline | dicefocal | 1 | 8 | sub-024 | rsub-024_ses-1_acq-petra_run-01_PDw.nii | 0.6820 | 7.28 | 0.5174 |
| a_baseline | dicefocal | 1 | 9 | sub-038 | rsub-038_ses-4_acq-petra_run-01_PDw.nii | 0.6883 | 5.10 | 0.5248 |
| a_baseline | dicefocal | 1 | 10 | sub-052 | rsub-052_ses-4_acq-petra_run-01_PDw.nii | 0.7467 | 4.90 | 0.5958 |
| a_baseline | dicefocal | 2 | 0 | sub-009 | rsub-009_ses-1_acq-petra_run-01_PDw.nii | 0.6260 | 3.74 | 0.4556 |
| a_baseline | dicefocal | 2 | 1 | sub-018 | rsub-018_ses-4_acq-petra_run-01_PDw.nii | 0.7755 | 187.39 | 0.6333 |
| a_baseline | dicefocal | 2 | 2 | sub-023 | rsub-023_ses-2_acq-petra_run-02_PDw.nii | 0.7518 | 5.20 | 0.6023 |
| a_baseline | dicefocal | 2 | 3 | sub-029 | rsub-029_ses-1_acq-petra_run-01_PDw.nii | 0.6712 | 112.66 | 0.5051 |
| a_baseline | dicefocal | 2 | 4 | sub-029 | rsub-029_ses-3_acq-petra_run-02_PDw.nii | 0.6479 | 39.01 | 0.4792 |
| a_baseline | dicefocal | 2 | 5 | sub-036 | rsub-036_ses-4_acq-petra_run-01_PDw.nii | 0.7498 | 5.48 | 0.5997 |
| a_baseline | dicefocal | 2 | 6 | sub-043 | rsub-043_ses-1_acq-petra_run-02_PDw.nii | 0.6983 | 79.11 | 0.5365 |
| a_baseline | dicefocal | 2 | 7 | sub-054 | rsub-054_ses-2_acq-petra_run-01_PDw.nii | 0.5034 | 198.16 | 0.3364 |
| a_baseline | dicefocal | 2 | 8 | sub-054 | rsub-054_ses-2_acq-petra_run-02_PDw.nii | 0.5503 | 3.74 | 0.3796 |
| a_baseline | dicefocal | 2 | 9 | sub-054 | rsub-054_ses-3_acq-petra_run-02_PDw.nii | 0.7367 | 5.39 | 0.5832 |
| a_baseline | dicefocal | 2 | 10 | sub-073 | rsub-073_ses-2_acq-petra_run-01_PDw.nii | 0.5957 | 212.70 | 0.4242 |
| a_baseline | dicefocal | 3 | 0 | sub-004 | rsub-004_ses-1_acq-petra_run-02_PDw.nii | 0.6100 | 11.09 | 0.4388 |
| a_baseline | dicefocal | 3 | 1 | sub-016 | rsub-016_ses-2_acq-petra_run-01_PDw.nii | 0.7251 | 17.09 | 0.5687 |
| a_baseline | dicefocal | 3 | 2 | sub-016 | rsub-016_ses-2_acq-petra_run-02_PDw.nii | 0.5669 | 3.00 | 0.3956 |
| a_baseline | dicefocal | 3 | 3 | sub-016 | rsub-016_ses-3_acq-petra_run-01_PDw.nii | 0.7445 | 154.49 | 0.5930 |
| a_baseline | dicefocal | 3 | 4 | sub-017 | rsub-017_ses-3_acq-petra_run-01_PDw.nii | 0.7305 | 6.71 | 0.5754 |
| a_baseline | dicefocal | 3 | 5 | sub-017 | rsub-017_ses-4_acq-petra_run-01_PDw.nii | 0.8159 | 23.37 | 0.6891 |
| a_baseline | dicefocal | 3 | 6 | sub-032 | rsub-032_ses-3_acq-petra_run-01_PDw.nii | 0.5565 | 179.08 | 0.3855 |
| a_baseline | dicefocal | 3 | 7 | sub-034 | rsub-034_ses-4_acq-petra_run-02_PDw.nii | 0.7376 | 4.00 | 0.5843 |
| a_baseline | dicefocal | 3 | 8 | sub-037 | rsub-037_ses-4_acq-petra_run-01_PDw.nii | 0.7468 | 235.32 | 0.5959 |
| a_baseline | dicefocal | 3 | 9 | sub-058 | rsub-058_ses-3_acq-petra_run-01_PDw.nii | 0.7683 | 115.44 | 0.6238 |
| a_baseline | dicefocal | 3 | 10 | sub-062 | rsub-062_ses-4_acq-petra_run-01_PDw.nii | 0.7655 | 3.32 | 0.6200 |
| a_baseline | dicefocal | 4 | 0 | sub-003 | rsub-003_ses-3_acq-petra_run-02_PDw.nii | 0.7517 | 4.12 | 0.6021 |
| a_baseline | dicefocal | 4 | 1 | sub-005 | rsub-005_ses-4_acq-petra_run-02_PDw.nii | 0.6882 | 4.58 | 0.5246 |
| a_baseline | dicefocal | 4 | 2 | sub-007 | rsub-007_ses-3_acq-petra_run-02_PDw.nii | 0.5739 | 58.83 | 0.4024 |
| a_baseline | dicefocal | 4 | 3 | sub-007 | rsub-007_ses-4_acq-petra_run-01_PDw.nii | 0.5956 | 185.76 | 0.4241 |
| a_baseline | dicefocal | 4 | 4 | sub-010 | rsub-010_ses-3_acq-petra_run-01_PDw.nii | 0.5414 | 64.04 | 0.3712 |
| a_baseline | dicefocal | 4 | 5 | sub-010 | rsub-010_ses-4_acq-petra_run-01_PDw.nii | 0.5923 | 3.74 | 0.4208 |
| a_baseline | dicefocal | 4 | 6 | sub-013 | rsub-013_ses-1_acq-petra_run-01_PDw.nii | 0.8403 | 181.64 | 0.7245 |
| a_baseline | dicefocal | 4 | 7 | sub-013 | rsub-013_ses-2_acq-petra_run-02_PDw.nii | 0.6029 | 199.28 | 0.4315 |
| a_baseline | dicefocal | 4 | 8 | sub-020 | rsub-020_ses-2_acq-petra_run-02_PDw.nii | 0.7221 | 4.24 | 0.5651 |
| a_baseline | dicefocal | 4 | 9 | sub-066 | rsub-066_ses-1_acq-petra_run-02_PDw.nii | 0.7984 | 4.69 | 0.6644 |
| b_no_attention | dicece | 0 | 0 | sub-011 | rsub-011_ses-3_acq-petra_run-01_PDw.nii | 0.6822 | 5.10 | 0.5177 |
| b_no_attention | dicece | 0 | 1 | sub-012 | rsub-012_ses-4_acq-petra_run-02_PDw.nii | 0.6540 | 6.78 | 0.4858 |
| b_no_attention | dicece | 0 | 2 | sub-014 | rsub-014_ses-4_acq-petra_run-02_PDw.nii | 0.6614 | 185.38 | 0.4941 |
| b_no_attention | dicece | 0 | 3 | sub-022 | rsub-022_ses-1_acq-petra_run-02_PDw.nii | 0.6949 | 118.08 | 0.5324 |
| b_no_attention | dicece | 0 | 4 | sub-022 | rsub-022_ses-2_acq-petra_run-02_PDw.nii | 0.4008 | 211.43 | 0.2506 |
| b_no_attention | dicece | 0 | 5 | sub-022 | rsub-022_ses-3_acq-petra_run-01_PDw.nii | 0.5682 | 8.77 | 0.3968 |
| b_no_attention | dicece | 0 | 6 | sub-022 | rsub-022_ses-4_acq-petra_run-01_PDw.nii | 0.6943 | 12.00 | 0.5318 |
| b_no_attention | dicece | 0 | 7 | sub-030 | rsub-030_ses-2_acq-petra_run-01_PDw.nii | 0.7188 | 4.12 | 0.5611 |
| b_no_attention | dicece | 0 | 8 | sub-030 | rsub-030_ses-3_acq-petra_run-02_PDw.nii | 0.7632 | 3.74 | 0.6170 |
| b_no_attention | dicece | 0 | 9 | sub-040 | rsub-040_ses-2_acq-petra_run-01_PDw.nii | 0.7150 | 3.00 | 0.5564 |
| b_no_attention | dicece | 0 | 10 | sub-044 | rsub-044_ses-2_acq-petra_run-02_PDw.nii | 0.7558 | 3.74 | 0.6075 |
| b_no_attention | dicece | 1 | 0 | sub-001 | rsub-001_ses-3_acq-petra_run-01_PDw.nii | 0.7517 | 6.71 | 0.6022 |
| b_no_attention | dicece | 1 | 1 | sub-001 | rsub-001_ses-4_acq-petra_run-02_PDw.nii | 0.0000 | 233.62 | 0.0000 |
| b_no_attention | dicece | 1 | 2 | sub-002 | rsub-002_ses-1_acq-petra_run-01_PDw.nii | 0.5762 | 3.46 | 0.4047 |
| b_no_attention | dicece | 1 | 3 | sub-002 | rsub-002_ses-1_acq-petra_run-02_PDw.nii | 0.3642 | 24.37 | 0.2226 |
| b_no_attention | dicece | 1 | 4 | sub-002 | rsub-002_ses-2_acq-petra_run-01_PDw.nii | 0.6374 | 74.35 | 0.4677 |
| b_no_attention | dicece | 1 | 5 | sub-006 | rsub-006_ses-2_acq-petra_run-01_PDw.nii | 0.5913 | 5.48 | 0.4198 |
| b_no_attention | dicece | 1 | 6 | sub-015 | rsub-015_ses-1_acq-petra_run-02_PDw.nii | 0.7117 | 6.00 | 0.5524 |
| b_no_attention | dicece | 1 | 7 | sub-021 | rsub-021_ses-3_acq-petra_run-01_PDw.nii | 0.6912 | 5.00 | 0.5281 |
| b_no_attention | dicece | 1 | 8 | sub-024 | rsub-024_ses-1_acq-petra_run-01_PDw.nii | 0.6966 | 5.00 | 0.5344 |
| b_no_attention | dicece | 1 | 9 | sub-038 | rsub-038_ses-4_acq-petra_run-01_PDw.nii | 0.7651 | 4.12 | 0.6195 |
| b_no_attention | dicece | 1 | 10 | sub-052 | rsub-052_ses-4_acq-petra_run-01_PDw.nii | 0.7678 | 5.00 | 0.6231 |
| b_no_attention | dicece | 2 | 0 | sub-009 | rsub-009_ses-1_acq-petra_run-01_PDw.nii | 0.6790 | 4.24 | 0.5140 |
| b_no_attention | dicece | 2 | 1 | sub-018 | rsub-018_ses-4_acq-petra_run-01_PDw.nii | 0.6389 | 29.56 | 0.4694 |
| b_no_attention | dicece | 2 | 2 | sub-023 | rsub-023_ses-2_acq-petra_run-02_PDw.nii | 0.6875 | 58.99 | 0.5238 |
| b_no_attention | dicece | 2 | 3 | sub-029 | rsub-029_ses-1_acq-petra_run-01_PDw.nii | 0.7336 | 175.82 | 0.5793 |
| b_no_attention | dicece | 2 | 4 | sub-029 | rsub-029_ses-3_acq-petra_run-02_PDw.nii | 0.5865 | 4.90 | 0.4149 |
| b_no_attention | dicece | 2 | 5 | sub-036 | rsub-036_ses-4_acq-petra_run-01_PDw.nii | 0.7741 | 5.00 | 0.6315 |
| b_no_attention | dicece | 2 | 6 | sub-043 | rsub-043_ses-1_acq-petra_run-02_PDw.nii | 0.5533 | 5.92 | 0.3824 |
| b_no_attention | dicece | 2 | 7 | sub-054 | rsub-054_ses-2_acq-petra_run-01_PDw.nii | 0.6586 | 46.11 | 0.4910 |
| b_no_attention | dicece | 2 | 8 | sub-054 | rsub-054_ses-2_acq-petra_run-02_PDw.nii | 0.6525 | 4.12 | 0.4842 |
| b_no_attention | dicece | 2 | 9 | sub-054 | rsub-054_ses-3_acq-petra_run-02_PDw.nii | 0.7497 | 7.07 | 0.5996 |
| b_no_attention | dicece | 2 | 10 | sub-073 | rsub-073_ses-2_acq-petra_run-01_PDw.nii | 0.7699 | 3.61 | 0.6259 |
| b_no_attention | dicece | 3 | 0 | sub-004 | rsub-004_ses-1_acq-petra_run-02_PDw.nii | 0.6296 | 4.90 | 0.4595 |
| b_no_attention | dicece | 3 | 1 | sub-016 | rsub-016_ses-2_acq-petra_run-01_PDw.nii | 0.7604 | 3.61 | 0.6134 |
| b_no_attention | dicece | 3 | 2 | sub-016 | rsub-016_ses-2_acq-petra_run-02_PDw.nii | 0.0220 | 107.09 | 0.0111 |
| b_no_attention | dicece | 3 | 3 | sub-016 | rsub-016_ses-3_acq-petra_run-01_PDw.nii | 0.7254 | 3.16 | 0.5691 |
| b_no_attention | dicece | 3 | 4 | sub-017 | rsub-017_ses-3_acq-petra_run-01_PDw.nii | 0.6971 | 3.61 | 0.5351 |
| b_no_attention | dicece | 3 | 5 | sub-017 | rsub-017_ses-4_acq-petra_run-01_PDw.nii | 0.7881 | 2.83 | 0.6502 |
| b_no_attention | dicece | 3 | 6 | sub-032 | rsub-032_ses-3_acq-petra_run-01_PDw.nii | 0.7185 | 5.83 | 0.5607 |
| b_no_attention | dicece | 3 | 7 | sub-034 | rsub-034_ses-4_acq-petra_run-02_PDw.nii | 0.7127 | 25.38 | 0.5536 |
| b_no_attention | dicece | 3 | 8 | sub-037 | rsub-037_ses-4_acq-petra_run-01_PDw.nii | 0.7073 | 5.92 | 0.5472 |
| b_no_attention | dicece | 3 | 9 | sub-058 | rsub-058_ses-3_acq-petra_run-01_PDw.nii | 0.6540 | 4.47 | 0.4858 |
| b_no_attention | dicece | 3 | 10 | sub-062 | rsub-062_ses-4_acq-petra_run-01_PDw.nii | 0.6598 | 3.32 | 0.4924 |
| b_no_attention | dicece | 4 | 0 | sub-003 | rsub-003_ses-3_acq-petra_run-02_PDw.nii | 0.7533 | 5.00 | 0.6042 |
| b_no_attention | dicece | 4 | 1 | sub-005 | rsub-005_ses-4_acq-petra_run-02_PDw.nii | 0.5796 | 48.08 | 0.4081 |
| b_no_attention | dicece | 4 | 2 | sub-007 | rsub-007_ses-3_acq-petra_run-02_PDw.nii | 0.4323 | 77.74 | 0.2758 |
| b_no_attention | dicece | 4 | 3 | sub-007 | rsub-007_ses-4_acq-petra_run-01_PDw.nii | 0.6327 | 4.58 | 0.4628 |
| b_no_attention | dicece | 4 | 4 | sub-010 | rsub-010_ses-3_acq-petra_run-01_PDw.nii | 0.6827 | 28.79 | 0.5183 |
| b_no_attention | dicece | 4 | 5 | sub-010 | rsub-010_ses-4_acq-petra_run-01_PDw.nii | 0.7321 | 4.58 | 0.5775 |
| b_no_attention | dicece | 4 | 6 | sub-013 | rsub-013_ses-1_acq-petra_run-01_PDw.nii | 0.0000 | 0.00 | 0.0000 |
| b_no_attention | dicece | 4 | 7 | sub-013 | rsub-013_ses-2_acq-petra_run-02_PDw.nii | 0.3626 | 22.65 | 0.2214 |
| b_no_attention | dicece | 4 | 8 | sub-020 | rsub-020_ses-2_acq-petra_run-02_PDw.nii | 0.7359 | 160.37 | 0.5822 |
| b_no_attention | dicece | 4 | 9 | sub-066 | rsub-066_ses-1_acq-petra_run-02_PDw.nii | 0.7050 | 4.24 | 0.5444 |
| b_no_attention | dicefocal | 0 | 0 | sub-011 | rsub-011_ses-3_acq-petra_run-01_PDw.nii | 0.7072 | 5.39 | 0.5471 |
| b_no_attention | dicefocal | 0 | 1 | sub-012 | rsub-012_ses-4_acq-petra_run-02_PDw.nii | 0.7425 | 4.24 | 0.5905 |
| b_no_attention | dicefocal | 0 | 2 | sub-014 | rsub-014_ses-4_acq-petra_run-02_PDw.nii | 0.6907 | 192.45 | 0.5276 |
| b_no_attention | dicefocal | 0 | 3 | sub-022 | rsub-022_ses-1_acq-petra_run-02_PDw.nii | 0.7281 | 3.16 | 0.5725 |
| b_no_attention | dicefocal | 0 | 4 | sub-022 | rsub-022_ses-2_acq-petra_run-02_PDw.nii | 0.6929 | 8.31 | 0.5301 |
| b_no_attention | dicefocal | 0 | 5 | sub-022 | rsub-022_ses-3_acq-petra_run-01_PDw.nii | 0.3189 | 26.87 | 0.1897 |
| b_no_attention | dicefocal | 0 | 6 | sub-022 | rsub-022_ses-4_acq-petra_run-01_PDw.nii | 0.5052 | 11.45 | 0.3380 |
| b_no_attention | dicefocal | 0 | 7 | sub-030 | rsub-030_ses-2_acq-petra_run-01_PDw.nii | 0.7325 | 12.81 | 0.5780 |
| b_no_attention | dicefocal | 0 | 8 | sub-030 | rsub-030_ses-3_acq-petra_run-02_PDw.nii | 0.7106 | 4.24 | 0.5511 |
| b_no_attention | dicefocal | 0 | 9 | sub-040 | rsub-040_ses-2_acq-petra_run-01_PDw.nii | 0.7270 | 10.05 | 0.5710 |
| b_no_attention | dicefocal | 0 | 10 | sub-044 | rsub-044_ses-2_acq-petra_run-02_PDw.nii | 0.7915 | 5.10 | 0.6550 |
| b_no_attention | dicefocal | 1 | 0 | sub-001 | rsub-001_ses-3_acq-petra_run-01_PDw.nii | 0.7574 | 6.00 | 0.6096 |
| b_no_attention | dicefocal | 1 | 1 | sub-001 | rsub-001_ses-4_acq-petra_run-02_PDw.nii | 0.5895 | 184.10 | 0.4180 |
| b_no_attention | dicefocal | 1 | 2 | sub-002 | rsub-002_ses-1_acq-petra_run-01_PDw.nii | 0.5317 | 220.69 | 0.3621 |
| b_no_attention | dicefocal | 1 | 3 | sub-002 | rsub-002_ses-1_acq-petra_run-02_PDw.nii | 0.6367 | 4.12 | 0.4670 |
| b_no_attention | dicefocal | 1 | 4 | sub-002 | rsub-002_ses-2_acq-petra_run-01_PDw.nii | 0.6683 | 4.69 | 0.5019 |
| b_no_attention | dicefocal | 1 | 5 | sub-006 | rsub-006_ses-2_acq-petra_run-01_PDw.nii | 0.6043 | 59.55 | 0.4329 |
| b_no_attention | dicefocal | 1 | 6 | sub-015 | rsub-015_ses-1_acq-petra_run-02_PDw.nii | 0.6779 | 5.48 | 0.5128 |
| b_no_attention | dicefocal | 1 | 7 | sub-021 | rsub-021_ses-3_acq-petra_run-01_PDw.nii | 0.6523 | 6.40 | 0.4840 |
| b_no_attention | dicefocal | 1 | 8 | sub-024 | rsub-024_ses-1_acq-petra_run-01_PDw.nii | 0.6631 | 6.16 | 0.4960 |
| b_no_attention | dicefocal | 1 | 9 | sub-038 | rsub-038_ses-4_acq-petra_run-01_PDw.nii | 0.7603 | 4.12 | 0.6133 |
| b_no_attention | dicefocal | 1 | 10 | sub-052 | rsub-052_ses-4_acq-petra_run-01_PDw.nii | 0.7255 | 9.11 | 0.5692 |
| b_no_attention | dicefocal | 2 | 0 | sub-009 | rsub-009_ses-1_acq-petra_run-01_PDw.nii | 0.6088 | 5.00 | 0.4376 |
| b_no_attention | dicefocal | 2 | 1 | sub-018 | rsub-018_ses-4_acq-petra_run-01_PDw.nii | 0.7343 | 5.66 | 0.5801 |
| b_no_attention | dicefocal | 2 | 2 | sub-023 | rsub-023_ses-2_acq-petra_run-02_PDw.nii | 0.6056 | 14.32 | 0.4343 |
| b_no_attention | dicefocal | 2 | 3 | sub-029 | rsub-029_ses-1_acq-petra_run-01_PDw.nii | 0.7483 | 5.20 | 0.5978 |
| b_no_attention | dicefocal | 2 | 4 | sub-029 | rsub-029_ses-3_acq-petra_run-02_PDw.nii | 0.6687 | 5.48 | 0.5022 |
| b_no_attention | dicefocal | 2 | 5 | sub-036 | rsub-036_ses-4_acq-petra_run-01_PDw.nii | 0.7634 | 4.90 | 0.6174 |
| b_no_attention | dicefocal | 2 | 6 | sub-043 | rsub-043_ses-1_acq-petra_run-02_PDw.nii | 0.5756 | 94.35 | 0.4041 |
| b_no_attention | dicefocal | 2 | 7 | sub-054 | rsub-054_ses-2_acq-petra_run-01_PDw.nii | 0.6664 | 7.35 | 0.4997 |
| b_no_attention | dicefocal | 2 | 8 | sub-054 | rsub-054_ses-2_acq-petra_run-02_PDw.nii | 0.5250 | 158.06 | 0.3559 |
| b_no_attention | dicefocal | 2 | 9 | sub-054 | rsub-054_ses-3_acq-petra_run-02_PDw.nii | 0.6781 | 5.83 | 0.5130 |
| b_no_attention | dicefocal | 2 | 10 | sub-073 | rsub-073_ses-2_acq-petra_run-01_PDw.nii | 0.7347 | 3.74 | 0.5807 |
| b_no_attention | dicefocal | 3 | 0 | sub-004 | rsub-004_ses-1_acq-petra_run-02_PDw.nii | 0.6136 | 12.08 | 0.4426 |
| b_no_attention | dicefocal | 3 | 1 | sub-016 | rsub-016_ses-2_acq-petra_run-01_PDw.nii | 0.7517 | 5.20 | 0.6022 |
| b_no_attention | dicefocal | 3 | 2 | sub-016 | rsub-016_ses-2_acq-petra_run-02_PDw.nii | 0.2513 | 49.83 | 0.1437 |
| b_no_attention | dicefocal | 3 | 3 | sub-016 | rsub-016_ses-3_acq-petra_run-01_PDw.nii | 0.6679 | 4.12 | 0.5014 |
| b_no_attention | dicefocal | 3 | 4 | sub-017 | rsub-017_ses-3_acq-petra_run-01_PDw.nii | 0.7413 | 5.66 | 0.5890 |
| b_no_attention | dicefocal | 3 | 5 | sub-017 | rsub-017_ses-4_acq-petra_run-01_PDw.nii | 0.7109 | 17.35 | 0.5515 |
| b_no_attention | dicefocal | 3 | 6 | sub-032 | rsub-032_ses-3_acq-petra_run-01_PDw.nii | 0.7198 | 8.31 | 0.5622 |
| b_no_attention | dicefocal | 3 | 7 | sub-034 | rsub-034_ses-4_acq-petra_run-02_PDw.nii | 0.6441 | 88.46 | 0.4751 |
| b_no_attention | dicefocal | 3 | 8 | sub-037 | rsub-037_ses-4_acq-petra_run-01_PDw.nii | 0.6851 | 5.00 | 0.5210 |
| b_no_attention | dicefocal | 3 | 9 | sub-058 | rsub-058_ses-3_acq-petra_run-01_PDw.nii | 0.7133 | 5.74 | 0.5544 |
| b_no_attention | dicefocal | 3 | 10 | sub-062 | rsub-062_ses-4_acq-petra_run-01_PDw.nii | 0.6817 | 3.00 | 0.5171 |
| b_no_attention | dicefocal | 4 | 0 | sub-003 | rsub-003_ses-3_acq-petra_run-02_PDw.nii | 0.6720 | 183.15 | 0.5060 |
| b_no_attention | dicefocal | 4 | 1 | sub-005 | rsub-005_ses-4_acq-petra_run-02_PDw.nii | 0.6757 | 41.59 | 0.5103 |
| b_no_attention | dicefocal | 4 | 2 | sub-007 | rsub-007_ses-3_acq-petra_run-02_PDw.nii | 0.6751 | 26.85 | 0.5096 |
| b_no_attention | dicefocal | 4 | 3 | sub-007 | rsub-007_ses-4_acq-petra_run-01_PDw.nii | 0.5766 | 5.10 | 0.4051 |
| b_no_attention | dicefocal | 4 | 4 | sub-010 | rsub-010_ses-3_acq-petra_run-01_PDw.nii | 0.6749 | 4.58 | 0.5093 |
| b_no_attention | dicefocal | 4 | 5 | sub-010 | rsub-010_ses-4_acq-petra_run-01_PDw.nii | 0.7218 | 40.50 | 0.5647 |
| b_no_attention | dicefocal | 4 | 6 | sub-013 | rsub-013_ses-1_acq-petra_run-01_PDw.nii | 0.6705 | 14.87 | 0.5043 |
| b_no_attention | dicefocal | 4 | 7 | sub-013 | rsub-013_ses-2_acq-petra_run-02_PDw.nii | 0.7511 | 5.39 | 0.6014 |
| b_no_attention | dicefocal | 4 | 8 | sub-020 | rsub-020_ses-2_acq-petra_run-02_PDw.nii | 0.3764 | 11.58 | 0.2318 |
| b_no_attention | dicefocal | 4 | 9 | sub-066 | rsub-066_ses-1_acq-petra_run-02_PDw.nii | 0.6934 | 5.66 | 0.5307 |
| c_reduced | dicece | 0 | 0 | sub-011 | rsub-011_ses-3_acq-petra_run-01_PDw.nii | 0.7569 | 4.90 | 0.6088 |
| c_reduced | dicece | 0 | 1 | sub-012 | rsub-012_ses-4_acq-petra_run-02_PDw.nii | 0.5646 | 250.08 | 0.3933 |
| c_reduced | dicece | 0 | 2 | sub-014 | rsub-014_ses-4_acq-petra_run-02_PDw.nii | 0.4025 | 42.72 | 0.2520 |
| c_reduced | dicece | 0 | 3 | sub-022 | rsub-022_ses-1_acq-petra_run-02_PDw.nii | 0.6690 | 3.74 | 0.5027 |
| c_reduced | dicece | 0 | 4 | sub-022 | rsub-022_ses-2_acq-petra_run-02_PDw.nii | 0.4548 | 4.69 | 0.2943 |
| c_reduced | dicece | 0 | 5 | sub-022 | rsub-022_ses-3_acq-petra_run-01_PDw.nii | 0.0000 | inf | 0.0000 |
| c_reduced | dicece | 0 | 6 | sub-022 | rsub-022_ses-4_acq-petra_run-01_PDw.nii | 0.4693 | 183.71 | 0.3066 |
| c_reduced | dicece | 0 | 7 | sub-030 | rsub-030_ses-2_acq-petra_run-01_PDw.nii | 0.7476 | 4.36 | 0.5969 |
| c_reduced | dicece | 0 | 8 | sub-030 | rsub-030_ses-3_acq-petra_run-02_PDw.nii | 0.6772 | 5.39 | 0.5119 |
| c_reduced | dicece | 0 | 9 | sub-040 | rsub-040_ses-2_acq-petra_run-01_PDw.nii | 0.6795 | 2.83 | 0.5146 |
| c_reduced | dicece | 0 | 10 | sub-044 | rsub-044_ses-2_acq-petra_run-02_PDw.nii | 0.7600 | 5.48 | 0.6129 |
| c_reduced | dicece | 1 | 0 | sub-001 | rsub-001_ses-3_acq-petra_run-01_PDw.nii | 0.7838 | 6.00 | 0.6444 |
| c_reduced | dicece | 1 | 1 | sub-001 | rsub-001_ses-4_acq-petra_run-02_PDw.nii | 0.6654 | 4.12 | 0.4985 |
| c_reduced | dicece | 1 | 2 | sub-002 | rsub-002_ses-1_acq-petra_run-01_PDw.nii | 0.6292 | 3.46 | 0.4590 |
| c_reduced | dicece | 1 | 3 | sub-002 | rsub-002_ses-1_acq-petra_run-02_PDw.nii | 0.6011 | 23.87 | 0.4297 |
| c_reduced | dicece | 1 | 4 | sub-002 | rsub-002_ses-2_acq-petra_run-01_PDw.nii | 0.0000 | 149.53 | 0.0000 |
| c_reduced | dicece | 1 | 5 | sub-006 | rsub-006_ses-2_acq-petra_run-01_PDw.nii | 0.6198 | 3.16 | 0.4491 |
| c_reduced | dicece | 1 | 6 | sub-015 | rsub-015_ses-1_acq-petra_run-02_PDw.nii | 0.6833 | 7.00 | 0.5189 |
| c_reduced | dicece | 1 | 7 | sub-021 | rsub-021_ses-3_acq-petra_run-01_PDw.nii | 0.7419 | 5.48 | 0.5897 |
| c_reduced | dicece | 1 | 8 | sub-024 | rsub-024_ses-1_acq-petra_run-01_PDw.nii | 0.6424 | 19.34 | 0.4731 |
| c_reduced | dicece | 1 | 9 | sub-038 | rsub-038_ses-4_acq-petra_run-01_PDw.nii | 0.7932 | 4.24 | 0.6573 |
| c_reduced | dicece | 1 | 10 | sub-052 | rsub-052_ses-4_acq-petra_run-01_PDw.nii | 0.6706 | 14.76 | 0.5044 |
| c_reduced | dicece | 2 | 0 | sub-009 | rsub-009_ses-1_acq-petra_run-01_PDw.nii | 0.6628 | 34.90 | 0.4957 |
| c_reduced | dicece | 2 | 1 | sub-018 | rsub-018_ses-4_acq-petra_run-01_PDw.nii | 0.7423 | 3.74 | 0.5902 |
| c_reduced | dicece | 2 | 2 | sub-023 | rsub-023_ses-2_acq-petra_run-02_PDw.nii | 0.7705 | 5.39 | 0.6267 |
| c_reduced | dicece | 2 | 3 | sub-029 | rsub-029_ses-1_acq-petra_run-01_PDw.nii | 0.5443 | 111.31 | 0.3739 |
| c_reduced | dicece | 2 | 4 | sub-029 | rsub-029_ses-3_acq-petra_run-02_PDw.nii | 0.6409 | 113.64 | 0.4716 |
| c_reduced | dicece | 2 | 5 | sub-036 | rsub-036_ses-4_acq-petra_run-01_PDw.nii | 0.7223 | 4.69 | 0.5653 |
| c_reduced | dicece | 2 | 6 | sub-043 | rsub-043_ses-1_acq-petra_run-02_PDw.nii | 0.7379 | 3.61 | 0.5847 |
| c_reduced | dicece | 2 | 7 | sub-054 | rsub-054_ses-2_acq-petra_run-01_PDw.nii | 0.5602 | 25.20 | 0.3891 |
| c_reduced | dicece | 2 | 8 | sub-054 | rsub-054_ses-2_acq-petra_run-02_PDw.nii | 0.4670 | 150.75 | 0.3047 |
| c_reduced | dicece | 2 | 9 | sub-054 | rsub-054_ses-3_acq-petra_run-02_PDw.nii | 0.5958 | 22.49 | 0.4243 |
| c_reduced | dicece | 2 | 10 | sub-073 | rsub-073_ses-2_acq-petra_run-01_PDw.nii | 0.6974 | 3.32 | 0.5354 |
| c_reduced | dicece | 3 | 0 | sub-004 | rsub-004_ses-1_acq-petra_run-02_PDw.nii | 0.6892 | 5.00 | 0.5257 |
| c_reduced | dicece | 3 | 1 | sub-016 | rsub-016_ses-2_acq-petra_run-01_PDw.nii | 0.6216 | 123.98 | 0.4509 |
| c_reduced | dicece | 3 | 2 | sub-016 | rsub-016_ses-2_acq-petra_run-02_PDw.nii | 0.6907 | 4.00 | 0.5275 |
| c_reduced | dicece | 3 | 3 | sub-016 | rsub-016_ses-3_acq-petra_run-01_PDw.nii | 0.7179 | 155.78 | 0.5599 |
| c_reduced | dicece | 3 | 4 | sub-017 | rsub-017_ses-3_acq-petra_run-01_PDw.nii | 0.7129 | 4.24 | 0.5539 |
| c_reduced | dicece | 3 | 5 | sub-017 | rsub-017_ses-4_acq-petra_run-01_PDw.nii | 0.7818 | 24.94 | 0.6417 |
| c_reduced | dicece | 3 | 6 | sub-032 | rsub-032_ses-3_acq-petra_run-01_PDw.nii | 0.7077 | 5.74 | 0.5476 |
| c_reduced | dicece | 3 | 7 | sub-034 | rsub-034_ses-4_acq-petra_run-02_PDw.nii | 0.6824 | 3.74 | 0.5179 |
| c_reduced | dicece | 3 | 8 | sub-037 | rsub-037_ses-4_acq-petra_run-01_PDw.nii | 0.6241 | 133.81 | 0.4536 |
| c_reduced | dicece | 3 | 9 | sub-058 | rsub-058_ses-3_acq-petra_run-01_PDw.nii | 0.7171 | 4.69 | 0.5590 |
| c_reduced | dicece | 3 | 10 | sub-062 | rsub-062_ses-4_acq-petra_run-01_PDw.nii | 0.6975 | 5.00 | 0.5355 |
| c_reduced | dicece | 4 | 0 | sub-003 | rsub-003_ses-3_acq-petra_run-02_PDw.nii | 0.6486 | 16.76 | 0.4799 |
| c_reduced | dicece | 4 | 1 | sub-005 | rsub-005_ses-4_acq-petra_run-02_PDw.nii | 0.6123 | 4.24 | 0.4412 |
| c_reduced | dicece | 4 | 2 | sub-007 | rsub-007_ses-3_acq-petra_run-02_PDw.nii | 0.4623 | 29.41 | 0.3007 |
| c_reduced | dicece | 4 | 3 | sub-007 | rsub-007_ses-4_acq-petra_run-01_PDw.nii | 0.5146 | 113.60 | 0.3465 |
| c_reduced | dicece | 4 | 4 | sub-010 | rsub-010_ses-3_acq-petra_run-01_PDw.nii | 0.6791 | 48.76 | 0.5141 |
| c_reduced | dicece | 4 | 5 | sub-010 | rsub-010_ses-4_acq-petra_run-01_PDw.nii | 0.6942 | 8.06 | 0.5317 |
| c_reduced | dicece | 4 | 6 | sub-013 | rsub-013_ses-1_acq-petra_run-01_PDw.nii | 0.3352 | 177.59 | 0.2014 |
| c_reduced | dicece | 4 | 7 | sub-013 | rsub-013_ses-2_acq-petra_run-02_PDw.nii | 0.7403 | 17.66 | 0.5877 |
| c_reduced | dicece | 4 | 8 | sub-020 | rsub-020_ses-2_acq-petra_run-02_PDw.nii | 0.4132 | 185.84 | 0.2604 |
| c_reduced | dicece | 4 | 9 | sub-066 | rsub-066_ses-1_acq-petra_run-02_PDw.nii | 0.2741 | 132.19 | 0.1588 |
| c_reduced | dicefocal | 0 | 0 | sub-011 | rsub-011_ses-3_acq-petra_run-01_PDw.nii | 0.7092 | 7.81 | 0.5494 |
| c_reduced | dicefocal | 0 | 1 | sub-012 | rsub-012_ses-4_acq-petra_run-02_PDw.nii | 0.0000 | 224.28 | 0.0000 |
| c_reduced | dicefocal | 0 | 2 | sub-014 | rsub-014_ses-4_acq-petra_run-02_PDw.nii | 0.6412 | 10.05 | 0.4719 |
| c_reduced | dicefocal | 0 | 3 | sub-022 | rsub-022_ses-1_acq-petra_run-02_PDw.nii | 0.7879 | 3.00 | 0.6500 |
| c_reduced | dicefocal | 0 | 4 | sub-022 | rsub-022_ses-2_acq-petra_run-02_PDw.nii | 0.6038 | 8.77 | 0.4325 |
| c_reduced | dicefocal | 0 | 5 | sub-022 | rsub-022_ses-3_acq-petra_run-01_PDw.nii | 0.5661 | 33.60 | 0.3948 |
| c_reduced | dicefocal | 0 | 6 | sub-022 | rsub-022_ses-4_acq-petra_run-01_PDw.nii | 0.5709 | 34.19 | 0.3995 |
| c_reduced | dicefocal | 0 | 7 | sub-030 | rsub-030_ses-2_acq-petra_run-01_PDw.nii | 0.6751 | 35.81 | 0.5095 |
| c_reduced | dicefocal | 0 | 8 | sub-030 | rsub-030_ses-3_acq-petra_run-02_PDw.nii | 0.7720 | 3.74 | 0.6286 |
| c_reduced | dicefocal | 0 | 9 | sub-040 | rsub-040_ses-2_acq-petra_run-01_PDw.nii | 0.7665 | 6.78 | 0.6213 |
| c_reduced | dicefocal | 0 | 10 | sub-044 | rsub-044_ses-2_acq-petra_run-02_PDw.nii | 0.7692 | 5.10 | 0.6250 |
| c_reduced | dicefocal | 1 | 0 | sub-001 | rsub-001_ses-3_acq-petra_run-01_PDw.nii | 0.7348 | 54.35 | 0.5808 |
| c_reduced | dicefocal | 1 | 1 | sub-001 | rsub-001_ses-4_acq-petra_run-02_PDw.nii | 0.6442 | 4.36 | 0.4752 |
| c_reduced | dicefocal | 1 | 2 | sub-002 | rsub-002_ses-1_acq-petra_run-01_PDw.nii | 0.5700 | 35.00 | 0.3986 |
| c_reduced | dicefocal | 1 | 3 | sub-002 | rsub-002_ses-1_acq-petra_run-02_PDw.nii | 0.4723 | 29.48 | 0.3091 |
| c_reduced | dicefocal | 1 | 4 | sub-002 | rsub-002_ses-2_acq-petra_run-01_PDw.nii | 0.6514 | 17.46 | 0.4830 |
| c_reduced | dicefocal | 1 | 5 | sub-006 | rsub-006_ses-2_acq-petra_run-01_PDw.nii | 0.6061 | 30.43 | 0.4348 |
| c_reduced | dicefocal | 1 | 6 | sub-015 | rsub-015_ses-1_acq-petra_run-02_PDw.nii | 0.7022 | 6.16 | 0.5411 |
| c_reduced | dicefocal | 1 | 7 | sub-021 | rsub-021_ses-3_acq-petra_run-01_PDw.nii | 0.6801 | 5.74 | 0.5153 |
| c_reduced | dicefocal | 1 | 8 | sub-024 | rsub-024_ses-1_acq-petra_run-01_PDw.nii | 0.7028 | 9.27 | 0.5418 |
| c_reduced | dicefocal | 1 | 9 | sub-038 | rsub-038_ses-4_acq-petra_run-01_PDw.nii | 0.5959 | 17.03 | 0.4244 |
| c_reduced | dicefocal | 1 | 10 | sub-052 | rsub-052_ses-4_acq-petra_run-01_PDw.nii | 0.7482 | 3.74 | 0.5977 |
| c_reduced | dicefocal | 2 | 0 | sub-009 | rsub-009_ses-1_acq-petra_run-01_PDw.nii | 0.5783 | 26.19 | 0.4068 |
| c_reduced | dicefocal | 2 | 1 | sub-018 | rsub-018_ses-4_acq-petra_run-01_PDw.nii | 0.7694 | 3.61 | 0.6252 |
| c_reduced | dicefocal | 2 | 2 | sub-023 | rsub-023_ses-2_acq-petra_run-02_PDw.nii | 0.7815 | 6.32 | 0.6414 |
| c_reduced | dicefocal | 2 | 3 | sub-029 | rsub-029_ses-1_acq-petra_run-01_PDw.nii | 0.6170 | 3.61 | 0.4461 |
| c_reduced | dicefocal | 2 | 4 | sub-029 | rsub-029_ses-3_acq-petra_run-02_PDw.nii | 0.4614 | 148.49 | 0.2999 |
| c_reduced | dicefocal | 2 | 5 | sub-036 | rsub-036_ses-4_acq-petra_run-01_PDw.nii | 0.7555 | 133.84 | 0.6071 |
| c_reduced | dicefocal | 2 | 6 | sub-043 | rsub-043_ses-1_acq-petra_run-02_PDw.nii | 0.7294 | 3.61 | 0.5740 |
| c_reduced | dicefocal | 2 | 7 | sub-054 | rsub-054_ses-2_acq-petra_run-01_PDw.nii | 0.0000 | 120.51 | 0.0000 |
| c_reduced | dicefocal | 2 | 8 | sub-054 | rsub-054_ses-2_acq-petra_run-02_PDw.nii | 0.0000 | 209.34 | 0.0000 |
| c_reduced | dicefocal | 2 | 9 | sub-054 | rsub-054_ses-3_acq-petra_run-02_PDw.nii | 0.6569 | 145.18 | 0.4891 |
| c_reduced | dicefocal | 2 | 10 | sub-073 | rsub-073_ses-2_acq-petra_run-01_PDw.nii | 0.7430 | 5.00 | 0.5911 |
| c_reduced | dicefocal | 4 | 0 | sub-003 | rsub-003_ses-3_acq-petra_run-02_PDw.nii | 0.4928 | 60.03 | 0.3270 |
| c_reduced | dicefocal | 4 | 1 | sub-005 | rsub-005_ses-4_acq-petra_run-02_PDw.nii | 0.5595 | 39.96 | 0.3884 |
| c_reduced | dicefocal | 4 | 2 | sub-007 | rsub-007_ses-3_acq-petra_run-02_PDw.nii | 0.4356 | 88.96 | 0.2784 |
| c_reduced | dicefocal | 4 | 3 | sub-007 | rsub-007_ses-4_acq-petra_run-01_PDw.nii | 0.5662 | 192.61 | 0.3949 |
| c_reduced | dicefocal | 4 | 4 | sub-010 | rsub-010_ses-3_acq-petra_run-01_PDw.nii | 0.6208 | 92.70 | 0.4501 |
| c_reduced | dicefocal | 4 | 5 | sub-010 | rsub-010_ses-4_acq-petra_run-01_PDw.nii | 0.1745 | 118.57 | 0.0956 |
| c_reduced | dicefocal | 4 | 6 | sub-013 | rsub-013_ses-1_acq-petra_run-01_PDw.nii | 0.0000 | inf | 0.0000 |
| c_reduced | dicefocal | 4 | 7 | sub-013 | rsub-013_ses-2_acq-petra_run-02_PDw.nii | 0.7272 | 6.08 | 0.5713 |
| c_reduced | dicefocal | 4 | 8 | sub-020 | rsub-020_ses-2_acq-petra_run-02_PDw.nii | 0.2581 | 51.80 | 0.1482 |
| c_reduced | dicefocal | 4 | 9 | sub-066 | rsub-066_ses-1_acq-petra_run-02_PDw.nii | 0.7395 | 4.00 | 0.5867 |
| d_increased | dicece | 0 | 0 | sub-011 | rsub-011_ses-3_acq-petra_run-01_PDw.nii | 0.7163 | 4.47 | 0.5580 |
| d_increased | dicece | 0 | 1 | sub-012 | rsub-012_ses-4_acq-petra_run-02_PDw.nii | 0.7345 | 4.47 | 0.5804 |
| d_increased | dicece | 0 | 2 | sub-014 | rsub-014_ses-4_acq-petra_run-02_PDw.nii | 0.6640 | 6.16 | 0.4970 |
| d_increased | dicece | 0 | 3 | sub-022 | rsub-022_ses-1_acq-petra_run-02_PDw.nii | 0.7935 | 4.36 | 0.6577 |
| d_increased | dicece | 0 | 4 | sub-022 | rsub-022_ses-2_acq-petra_run-02_PDw.nii | 0.2077 | 208.95 | 0.1159 |
| d_increased | dicece | 0 | 5 | sub-022 | rsub-022_ses-3_acq-petra_run-01_PDw.nii | 0.5887 | 164.72 | 0.4172 |
| d_increased | dicece | 0 | 6 | sub-022 | rsub-022_ses-4_acq-petra_run-01_PDw.nii | 0.6648 | 9.17 | 0.4980 |
| d_increased | dicece | 0 | 7 | sub-030 | rsub-030_ses-2_acq-petra_run-01_PDw.nii | 0.7506 | 3.61 | 0.6007 |
| d_increased | dicece | 0 | 8 | sub-030 | rsub-030_ses-3_acq-petra_run-02_PDw.nii | 0.6089 | 12.08 | 0.4377 |
| d_increased | dicece | 0 | 9 | sub-040 | rsub-040_ses-2_acq-petra_run-01_PDw.nii | 0.8030 | 4.12 | 0.6708 |
| d_increased | dicece | 0 | 10 | sub-044 | rsub-044_ses-2_acq-petra_run-02_PDw.nii | 0.7844 | 4.58 | 0.6453 |
| d_increased | dicece | 1 | 0 | sub-001 | rsub-001_ses-3_acq-petra_run-01_PDw.nii | 0.3097 | 153.70 | 0.1832 |
| d_increased | dicece | 1 | 1 | sub-001 | rsub-001_ses-4_acq-petra_run-02_PDw.nii | 0.6440 | 4.24 | 0.4749 |
| d_increased | dicece | 1 | 2 | sub-002 | rsub-002_ses-1_acq-petra_run-01_PDw.nii | 0.6158 | 4.47 | 0.4449 |
| d_increased | dicece | 1 | 3 | sub-002 | rsub-002_ses-1_acq-petra_run-02_PDw.nii | 0.5432 | 4.24 | 0.3729 |
| d_increased | dicece | 1 | 4 | sub-002 | rsub-002_ses-2_acq-petra_run-01_PDw.nii | 0.6800 | 3.74 | 0.5151 |
| d_increased | dicece | 1 | 5 | sub-006 | rsub-006_ses-2_acq-petra_run-01_PDw.nii | 0.6655 | 3.00 | 0.4986 |
| d_increased | dicece | 1 | 6 | sub-015 | rsub-015_ses-1_acq-petra_run-02_PDw.nii | 0.6359 | 35.06 | 0.4661 |
| d_increased | dicece | 1 | 7 | sub-021 | rsub-021_ses-3_acq-petra_run-01_PDw.nii | 0.7226 | 4.12 | 0.5657 |
| d_increased | dicece | 1 | 8 | sub-024 | rsub-024_ses-1_acq-petra_run-01_PDw.nii | 0.6858 | 7.68 | 0.5218 |
| d_increased | dicece | 1 | 9 | sub-038 | rsub-038_ses-4_acq-petra_run-01_PDw.nii | 0.7796 | 4.12 | 0.6388 |
| d_increased | dicece | 1 | 10 | sub-052 | rsub-052_ses-4_acq-petra_run-01_PDw.nii | 0.6611 | 54.00 | 0.4938 |
| d_increased | dicece | 2 | 0 | sub-009 | rsub-009_ses-1_acq-petra_run-01_PDw.nii | 0.6395 | 3.16 | 0.4701 |
| d_increased | dicece | 2 | 1 | sub-018 | rsub-018_ses-4_acq-petra_run-01_PDw.nii | 0.7159 | 6.48 | 0.5575 |
| d_increased | dicece | 2 | 2 | sub-023 | rsub-023_ses-2_acq-petra_run-02_PDw.nii | 0.7484 | 3.16 | 0.5980 |
| d_increased | dicece | 2 | 3 | sub-029 | rsub-029_ses-1_acq-petra_run-01_PDw.nii | 0.6544 | 4.12 | 0.4863 |
| d_increased | dicece | 2 | 4 | sub-029 | rsub-029_ses-3_acq-petra_run-02_PDw.nii | 0.5915 | 117.75 | 0.4199 |
| d_increased | dicece | 2 | 5 | sub-036 | rsub-036_ses-4_acq-petra_run-01_PDw.nii | 0.6971 | 6.48 | 0.5351 |
| d_increased | dicece | 2 | 6 | sub-043 | rsub-043_ses-1_acq-petra_run-02_PDw.nii | 0.7530 | 3.74 | 0.6038 |
| d_increased | dicece | 2 | 7 | sub-054 | rsub-054_ses-2_acq-petra_run-01_PDw.nii | 0.1540 | 41.63 | 0.0835 |
| d_increased | dicece | 2 | 8 | sub-054 | rsub-054_ses-2_acq-petra_run-02_PDw.nii | 0.6853 | 9.38 | 0.5212 |
| d_increased | dicece | 2 | 9 | sub-054 | rsub-054_ses-3_acq-petra_run-02_PDw.nii | 0.6531 | 5.66 | 0.4849 |
| d_increased | dicece | 2 | 10 | sub-073 | rsub-073_ses-2_acq-petra_run-01_PDw.nii | 0.7541 | 24.52 | 0.6053 |
| d_increased | dicece | 3 | 0 | sub-004 | rsub-004_ses-1_acq-petra_run-02_PDw.nii | 0.6614 | 5.10 | 0.4941 |
| d_increased | dicece | 3 | 1 | sub-016 | rsub-016_ses-2_acq-petra_run-01_PDw.nii | 0.7739 | 5.74 | 0.6312 |
| d_increased | dicece | 3 | 2 | sub-016 | rsub-016_ses-2_acq-petra_run-02_PDw.nii | 0.6668 | 7.07 | 0.5001 |
| d_increased | dicece | 3 | 3 | sub-016 | rsub-016_ses-3_acq-petra_run-01_PDw.nii | 0.6889 | 5.20 | 0.5255 |
| d_increased | dicece | 3 | 4 | sub-017 | rsub-017_ses-3_acq-petra_run-01_PDw.nii | 0.6828 | 14.49 | 0.5183 |
| d_increased | dicece | 3 | 5 | sub-017 | rsub-017_ses-4_acq-petra_run-01_PDw.nii | 0.7303 | 26.94 | 0.5751 |
| d_increased | dicece | 3 | 6 | sub-032 | rsub-032_ses-3_acq-petra_run-01_PDw.nii | 0.6784 | 19.95 | 0.5133 |
| d_increased | dicece | 3 | 7 | sub-034 | rsub-034_ses-4_acq-petra_run-02_PDw.nii | 0.7276 | 5.39 | 0.5718 |
| d_increased | dicece | 3 | 8 | sub-037 | rsub-037_ses-4_acq-petra_run-01_PDw.nii | 0.4986 | 155.97 | 0.3321 |
| d_increased | dicece | 3 | 9 | sub-058 | rsub-058_ses-3_acq-petra_run-01_PDw.nii | 0.7513 | 138.71 | 0.6017 |
| d_increased | dicece | 3 | 10 | sub-062 | rsub-062_ses-4_acq-petra_run-01_PDw.nii | 0.6680 | 6.00 | 0.5016 |
| d_increased | dicece | 4 | 0 | sub-003 | rsub-003_ses-3_acq-petra_run-02_PDw.nii | 0.7306 | 3.61 | 0.5756 |
| d_increased | dicece | 4 | 1 | sub-005 | rsub-005_ses-4_acq-petra_run-02_PDw.nii | 0.6726 | 5.48 | 0.5067 |
| d_increased | dicece | 4 | 2 | sub-007 | rsub-007_ses-3_acq-petra_run-02_PDw.nii | 0.0000 | 251.25 | 0.0000 |
| d_increased | dicece | 4 | 3 | sub-007 | rsub-007_ses-4_acq-petra_run-01_PDw.nii | 0.5550 | 21.21 | 0.3841 |
| d_increased | dicece | 4 | 4 | sub-010 | rsub-010_ses-3_acq-petra_run-01_PDw.nii | 0.5303 | 25.48 | 0.3608 |
| d_increased | dicece | 4 | 5 | sub-010 | rsub-010_ses-4_acq-petra_run-01_PDw.nii | 0.7483 | 5.10 | 0.5978 |
| d_increased | dicece | 4 | 6 | sub-013 | rsub-013_ses-1_acq-petra_run-01_PDw.nii | 0.4919 | 14.56 | 0.3262 |
| d_increased | dicece | 4 | 7 | sub-013 | rsub-013_ses-2_acq-petra_run-02_PDw.nii | 0.6735 | 5.39 | 0.5078 |
| d_increased | dicece | 4 | 8 | sub-020 | rsub-020_ses-2_acq-petra_run-02_PDw.nii | 0.7115 | 4.90 | 0.5522 |
| d_increased | dicece | 4 | 9 | sub-066 | rsub-066_ses-1_acq-petra_run-02_PDw.nii | 0.7804 | 4.90 | 0.6399 |
| d_increased | dicefocal | 0 | 0 | sub-011 | rsub-011_ses-3_acq-petra_run-01_PDw.nii | 0.0000 | 154.10 | 0.0000 |
| d_increased | dicefocal | 0 | 1 | sub-012 | rsub-012_ses-4_acq-petra_run-02_PDw.nii | 0.7081 | 3.61 | 0.5481 |
| d_increased | dicefocal | 0 | 2 | sub-014 | rsub-014_ses-4_acq-petra_run-02_PDw.nii | 0.6050 | 103.00 | 0.4337 |
| d_increased | dicefocal | 0 | 3 | sub-022 | rsub-022_ses-1_acq-petra_run-02_PDw.nii | 0.7028 | 164.68 | 0.5417 |
| d_increased | dicefocal | 0 | 4 | sub-022 | rsub-022_ses-2_acq-petra_run-02_PDw.nii | 0.0000 | inf | 0.0000 |
| d_increased | dicefocal | 0 | 5 | sub-022 | rsub-022_ses-3_acq-petra_run-01_PDw.nii | 0.3818 | 162.51 | 0.2359 |
| d_increased | dicefocal | 0 | 6 | sub-022 | rsub-022_ses-4_acq-petra_run-01_PDw.nii | 0.2946 | 217.50 | 0.1727 |
| d_increased | dicefocal | 0 | 7 | sub-030 | rsub-030_ses-2_acq-petra_run-01_PDw.nii | 0.7014 | 212.17 | 0.5401 |
| d_increased | dicefocal | 0 | 8 | sub-030 | rsub-030_ses-3_acq-petra_run-02_PDw.nii | 0.7701 | 2.83 | 0.6261 |
| d_increased | dicefocal | 0 | 9 | sub-040 | rsub-040_ses-2_acq-petra_run-01_PDw.nii | 0.7740 | 6.63 | 0.6313 |
| d_increased | dicefocal | 0 | 10 | sub-044 | rsub-044_ses-2_acq-petra_run-02_PDw.nii | 0.7355 | 5.48 | 0.5817 |
| d_increased | dicefocal | 1 | 0 | sub-001 | rsub-001_ses-3_acq-petra_run-01_PDw.nii | 0.7343 | 7.00 | 0.5802 |
| d_increased | dicefocal | 1 | 1 | sub-001 | rsub-001_ses-4_acq-petra_run-02_PDw.nii | 0.6478 | 177.79 | 0.4791 |
| d_increased | dicefocal | 1 | 2 | sub-002 | rsub-002_ses-1_acq-petra_run-01_PDw.nii | 0.5643 | 3.74 | 0.3931 |
| d_increased | dicefocal | 1 | 3 | sub-002 | rsub-002_ses-1_acq-petra_run-02_PDw.nii | 0.5555 | 4.24 | 0.3846 |
| d_increased | dicefocal | 1 | 4 | sub-002 | rsub-002_ses-2_acq-petra_run-01_PDw.nii | 0.4333 | 226.54 | 0.2765 |
| d_increased | dicefocal | 1 | 5 | sub-006 | rsub-006_ses-2_acq-petra_run-01_PDw.nii | 0.5865 | 55.74 | 0.4149 |
| d_increased | dicefocal | 1 | 6 | sub-015 | rsub-015_ses-1_acq-petra_run-02_PDw.nii | 0.0000 | inf | 0.0000 |
| d_increased | dicefocal | 1 | 7 | sub-021 | rsub-021_ses-3_acq-petra_run-01_PDw.nii | 0.7378 | 38.00 | 0.5845 |
| d_increased | dicefocal | 1 | 8 | sub-024 | rsub-024_ses-1_acq-petra_run-01_PDw.nii | 0.7082 | 5.10 | 0.5482 |
| d_increased | dicefocal | 1 | 9 | sub-038 | rsub-038_ses-4_acq-petra_run-01_PDw.nii | 0.5904 | 100.72 | 0.4188 |
| d_increased | dicefocal | 1 | 10 | sub-052 | rsub-052_ses-4_acq-petra_run-01_PDw.nii | 0.7380 | 7.28 | 0.5848 |
| d_increased | dicefocal | 2 | 0 | sub-009 | rsub-009_ses-1_acq-petra_run-01_PDw.nii | 0.7040 | 4.69 | 0.5433 |
| d_increased | dicefocal | 2 | 1 | sub-018 | rsub-018_ses-4_acq-petra_run-01_PDw.nii | 0.7569 | 3.74 | 0.6089 |
| d_increased | dicefocal | 2 | 2 | sub-023 | rsub-023_ses-2_acq-petra_run-02_PDw.nii | 0.7461 | 3.61 | 0.5950 |
| d_increased | dicefocal | 2 | 3 | sub-029 | rsub-029_ses-1_acq-petra_run-01_PDw.nii | 0.1627 | 98.30 | 0.0886 |
| d_increased | dicefocal | 2 | 4 | sub-029 | rsub-029_ses-3_acq-petra_run-02_PDw.nii | 0.6296 | 96.22 | 0.4594 |
| d_increased | dicefocal | 2 | 5 | sub-036 | rsub-036_ses-4_acq-petra_run-01_PDw.nii | 0.7245 | 4.47 | 0.5680 |
| d_increased | dicefocal | 2 | 6 | sub-043 | rsub-043_ses-1_acq-petra_run-02_PDw.nii | 0.5710 | 232.86 | 0.3996 |
| d_increased | dicefocal | 2 | 7 | sub-054 | rsub-054_ses-2_acq-petra_run-01_PDw.nii | 0.0309 | 30.43 | 0.0157 |
| d_increased | dicefocal | 2 | 8 | sub-054 | rsub-054_ses-2_acq-petra_run-02_PDw.nii | 0.0511 | 127.18 | 0.0262 |
| d_increased | dicefocal | 2 | 9 | sub-054 | rsub-054_ses-3_acq-petra_run-02_PDw.nii | 0.6797 | 4.90 | 0.5148 |
| d_increased | dicefocal | 2 | 10 | sub-073 | rsub-073_ses-2_acq-petra_run-01_PDw.nii | 0.7784 | 4.12 | 0.6372 |
| d_increased | dicefocal | 3 | 0 | sub-004 | rsub-004_ses-1_acq-petra_run-02_PDw.nii | 0.6699 | 4.12 | 0.5036 |
| d_increased | dicefocal | 3 | 1 | sub-016 | rsub-016_ses-2_acq-petra_run-01_PDw.nii | 0.6887 | 5.20 | 0.5253 |
| d_increased | dicefocal | 3 | 2 | sub-016 | rsub-016_ses-2_acq-petra_run-02_PDw.nii | 0.6599 | 3.00 | 0.4925 |
| d_increased | dicefocal | 3 | 3 | sub-016 | rsub-016_ses-3_acq-petra_run-01_PDw.nii | 0.6893 | 187.14 | 0.5259 |
| d_increased | dicefocal | 3 | 4 | sub-017 | rsub-017_ses-3_acq-petra_run-01_PDw.nii | 0.5611 | 31.62 | 0.3900 |
| d_increased | dicefocal | 3 | 5 | sub-017 | rsub-017_ses-4_acq-petra_run-01_PDw.nii | 0.7479 | 30.23 | 0.5974 |
| d_increased | dicefocal | 3 | 6 | sub-032 | rsub-032_ses-3_acq-petra_run-01_PDw.nii | 0.7350 | 8.31 | 0.5810 |
| d_increased | dicefocal | 3 | 7 | sub-034 | rsub-034_ses-4_acq-petra_run-02_PDw.nii | 0.7106 | 3.16 | 0.5511 |
| d_increased | dicefocal | 3 | 8 | sub-037 | rsub-037_ses-4_acq-petra_run-01_PDw.nii | 0.6916 | 187.70 | 0.5286 |
| d_increased | dicefocal | 3 | 9 | sub-058 | rsub-058_ses-3_acq-petra_run-01_PDw.nii | 0.7557 | 5.00 | 0.6073 |
| d_increased | dicefocal | 3 | 10 | sub-062 | rsub-062_ses-4_acq-petra_run-01_PDw.nii | 0.6975 | 4.12 | 0.5355 |
| d_increased | dicefocal | 4 | 0 | sub-003 | rsub-003_ses-3_acq-petra_run-02_PDw.nii | 0.7585 | 3.74 | 0.6110 |
| d_increased | dicefocal | 4 | 1 | sub-005 | rsub-005_ses-4_acq-petra_run-02_PDw.nii | 0.6194 | 6.48 | 0.4486 |
| d_increased | dicefocal | 4 | 2 | sub-007 | rsub-007_ses-3_acq-petra_run-02_PDw.nii | 0.5281 | 14.04 | 0.3588 |
| d_increased | dicefocal | 4 | 3 | sub-007 | rsub-007_ses-4_acq-petra_run-01_PDw.nii | 0.5100 | 5.66 | 0.3423 |
| d_increased | dicefocal | 4 | 4 | sub-010 | rsub-010_ses-3_acq-petra_run-01_PDw.nii | 0.5989 | 124.47 | 0.4274 |
| d_increased | dicefocal | 4 | 5 | sub-010 | rsub-010_ses-4_acq-petra_run-01_PDw.nii | 0.6968 | 6.63 | 0.5347 |
| d_increased | dicefocal | 4 | 6 | sub-013 | rsub-013_ses-1_acq-petra_run-01_PDw.nii | 0.7141 | 37.39 | 0.5553 |
| d_increased | dicefocal | 4 | 7 | sub-013 | rsub-013_ses-2_acq-petra_run-02_PDw.nii | 0.7213 | 5.20 | 0.5641 |
| d_increased | dicefocal | 4 | 8 | sub-020 | rsub-020_ses-2_acq-petra_run-02_PDw.nii | 0.7568 | 29.22 | 0.6087 |
| d_increased | dicefocal | 4 | 9 | sub-066 | rsub-066_ses-1_acq-petra_run-02_PDw.nii | 0.7248 | 152.11 | 0.5684 |
| proposed | dicece | 0 | 0 | sub-011 | rsub-011_ses-3_acq-petra_run-01_PDw.nii | 0.7275 | 4.58 | 0.5717 |
| proposed | dicece | 0 | 1 | sub-012 | rsub-012_ses-4_acq-petra_run-02_PDw.nii | 0.6825 | 136.17 | 0.5180 |
| proposed | dicece | 0 | 2 | sub-014 | rsub-014_ses-4_acq-petra_run-02_PDw.nii | 0.6317 | 18.06 | 0.4617 |
| proposed | dicece | 0 | 3 | sub-022 | rsub-022_ses-1_acq-petra_run-02_PDw.nii | 0.5606 | 17.15 | 0.3894 |
| proposed | dicece | 0 | 4 | sub-022 | rsub-022_ses-2_acq-petra_run-02_PDw.nii | 0.6484 | 5.20 | 0.4798 |
| proposed | dicece | 0 | 5 | sub-022 | rsub-022_ses-3_acq-petra_run-01_PDw.nii | 0.4793 | 28.23 | 0.3152 |
| proposed | dicece | 0 | 6 | sub-022 | rsub-022_ses-4_acq-petra_run-01_PDw.nii | 0.5737 | 162.66 | 0.4022 |
| proposed | dicece | 0 | 7 | sub-030 | rsub-030_ses-2_acq-petra_run-01_PDw.nii | 0.7594 | 4.36 | 0.6121 |
| proposed | dicece | 0 | 8 | sub-030 | rsub-030_ses-3_acq-petra_run-02_PDw.nii | 0.7543 | 3.00 | 0.6055 |
| proposed | dicece | 0 | 9 | sub-040 | rsub-040_ses-2_acq-petra_run-01_PDw.nii | 0.7275 | 10.72 | 0.5717 |
| proposed | dicece | 0 | 10 | sub-044 | rsub-044_ses-2_acq-petra_run-02_PDw.nii | 0.7586 | 3.32 | 0.6111 |
| proposed | dicece | 1 | 0 | sub-001 | rsub-001_ses-3_acq-petra_run-01_PDw.nii | 0.2113 | 231.31 | 0.1181 |
| proposed | dicece | 1 | 1 | sub-001 | rsub-001_ses-4_acq-petra_run-02_PDw.nii | 0.6228 | 9.49 | 0.4522 |
| proposed | dicece | 1 | 2 | sub-002 | rsub-002_ses-1_acq-petra_run-01_PDw.nii | 0.7249 | 32.16 | 0.5685 |
| proposed | dicece | 1 | 3 | sub-002 | rsub-002_ses-1_acq-petra_run-02_PDw.nii | 0.6381 | 4.12 | 0.4685 |
| proposed | dicece | 1 | 4 | sub-002 | rsub-002_ses-2_acq-petra_run-01_PDw.nii | 0.6696 | 5.48 | 0.5033 |
| proposed | dicece | 1 | 5 | sub-006 | rsub-006_ses-2_acq-petra_run-01_PDw.nii | 0.7176 | 5.74 | 0.5596 |
| proposed | dicece | 1 | 6 | sub-015 | rsub-015_ses-1_acq-petra_run-02_PDw.nii | 0.6265 | 11.83 | 0.4562 |
| proposed | dicece | 1 | 7 | sub-021 | rsub-021_ses-3_acq-petra_run-01_PDw.nii | 0.7592 | 4.90 | 0.6119 |
| proposed | dicece | 1 | 8 | sub-024 | rsub-024_ses-1_acq-petra_run-01_PDw.nii | 0.5820 | 115.26 | 0.4105 |
| proposed | dicece | 1 | 9 | sub-038 | rsub-038_ses-4_acq-petra_run-01_PDw.nii | 0.7877 | 4.12 | 0.6498 |
| proposed | dicece | 1 | 10 | sub-052 | rsub-052_ses-4_acq-petra_run-01_PDw.nii | 0.7572 | 5.00 | 0.6092 |
| proposed | dicece | 2 | 0 | sub-009 | rsub-009_ses-1_acq-petra_run-01_PDw.nii | 0.5959 | 3.61 | 0.4244 |
| proposed | dicece | 2 | 1 | sub-018 | rsub-018_ses-4_acq-petra_run-01_PDw.nii | 0.7592 | 4.00 | 0.6119 |
| proposed | dicece | 2 | 2 | sub-023 | rsub-023_ses-2_acq-petra_run-02_PDw.nii | 0.6390 | 114.78 | 0.4695 |
| proposed | dicece | 2 | 3 | sub-029 | rsub-029_ses-1_acq-petra_run-01_PDw.nii | 0.7265 | 3.46 | 0.5705 |
| proposed | dicece | 2 | 4 | sub-029 | rsub-029_ses-3_acq-petra_run-02_PDw.nii | 0.3181 | 261.55 | 0.1891 |
| proposed | dicece | 2 | 5 | sub-036 | rsub-036_ses-4_acq-petra_run-01_PDw.nii | 0.7463 | 6.40 | 0.5953 |
| proposed | dicece | 2 | 6 | sub-043 | rsub-043_ses-1_acq-petra_run-02_PDw.nii | 0.7083 | 2.83 | 0.5483 |
| proposed | dicece | 2 | 7 | sub-054 | rsub-054_ses-2_acq-petra_run-01_PDw.nii | 0.1902 | 36.73 | 0.1051 |
| proposed | dicece | 2 | 8 | sub-054 | rsub-054_ses-2_acq-petra_run-02_PDw.nii | 0.0000 | inf | 0.0000 |
| proposed | dicece | 2 | 9 | sub-054 | rsub-054_ses-3_acq-petra_run-02_PDw.nii | 0.6623 | 3.16 | 0.4951 |
| proposed | dicece | 2 | 10 | sub-073 | rsub-073_ses-2_acq-petra_run-01_PDw.nii | 0.7020 | 3.00 | 0.5409 |
| proposed | dicece | 3 | 0 | sub-004 | rsub-004_ses-1_acq-petra_run-02_PDw.nii | 0.6949 | 10.82 | 0.5325 |
| proposed | dicece | 3 | 1 | sub-016 | rsub-016_ses-2_acq-petra_run-01_PDw.nii | 0.7598 | 3.32 | 0.6126 |
| proposed | dicece | 3 | 2 | sub-016 | rsub-016_ses-2_acq-petra_run-02_PDw.nii | 0.6749 | 4.36 | 0.5094 |
| proposed | dicece | 3 | 3 | sub-016 | rsub-016_ses-3_acq-petra_run-01_PDw.nii | 0.6946 | 3.74 | 0.5321 |
| proposed | dicece | 3 | 4 | sub-017 | rsub-017_ses-3_acq-petra_run-01_PDw.nii | 0.6536 | 9.49 | 0.4854 |
| proposed | dicece | 3 | 5 | sub-017 | rsub-017_ses-4_acq-petra_run-01_PDw.nii | 0.7853 | 18.36 | 0.6465 |
| proposed | dicece | 3 | 6 | sub-032 | rsub-032_ses-3_acq-petra_run-01_PDw.nii | 0.5587 | 109.27 | 0.3877 |
| proposed | dicece | 3 | 7 | sub-034 | rsub-034_ses-4_acq-petra_run-02_PDw.nii | 0.6692 | 10.05 | 0.5029 |
| proposed | dicece | 3 | 8 | sub-037 | rsub-037_ses-4_acq-petra_run-01_PDw.nii | 0.7361 | 4.36 | 0.5825 |
| proposed | dicece | 3 | 9 | sub-058 | rsub-058_ses-3_acq-petra_run-01_PDw.nii | 0.7810 | 5.10 | 0.6407 |
| proposed | dicece | 3 | 10 | sub-062 | rsub-062_ses-4_acq-petra_run-01_PDw.nii | 0.5932 | 116.92 | 0.4217 |
| proposed | dicece | 4 | 0 | sub-003 | rsub-003_ses-3_acq-petra_run-02_PDw.nii | 0.3161 | 133.69 | 0.1877 |
| proposed | dicece | 4 | 1 | sub-005 | rsub-005_ses-4_acq-petra_run-02_PDw.nii | 0.7104 | 123.64 | 0.5509 |
| proposed | dicece | 4 | 2 | sub-007 | rsub-007_ses-3_acq-petra_run-02_PDw.nii | 0.5268 | 50.09 | 0.3576 |
| proposed | dicece | 4 | 3 | sub-007 | rsub-007_ses-4_acq-petra_run-01_PDw.nii | 0.6256 | 5.10 | 0.4552 |
| proposed | dicece | 4 | 4 | sub-010 | rsub-010_ses-3_acq-petra_run-01_PDw.nii | 0.6863 | 3.16 | 0.5224 |
| proposed | dicece | 4 | 5 | sub-010 | rsub-010_ses-4_acq-petra_run-01_PDw.nii | 0.7556 | 4.24 | 0.6072 |
| proposed | dicece | 4 | 6 | sub-013 | rsub-013_ses-1_acq-petra_run-01_PDw.nii | 0.6318 | 5.39 | 0.4617 |
| proposed | dicece | 4 | 7 | sub-013 | rsub-013_ses-2_acq-petra_run-02_PDw.nii | 0.5757 | 24.92 | 0.4042 |
| proposed | dicece | 4 | 8 | sub-020 | rsub-020_ses-2_acq-petra_run-02_PDw.nii | 0.7728 | 5.74 | 0.6298 |
| proposed | dicece | 4 | 9 | sub-066 | rsub-066_ses-1_acq-petra_run-02_PDw.nii | 0.7344 | 4.24 | 0.5803 |
| proposed | dicefocal | 0 | 0 | sub-011 | rsub-011_ses-3_acq-petra_run-01_PDw.nii | 0.7326 | 3.74 | 0.5781 |
| proposed | dicefocal | 0 | 1 | sub-012 | rsub-012_ses-4_acq-petra_run-02_PDw.nii | 0.7252 | 3.74 | 0.5689 |
| proposed | dicefocal | 0 | 2 | sub-014 | rsub-014_ses-4_acq-petra_run-02_PDw.nii | 0.6759 | 8.66 | 0.5105 |
| proposed | dicefocal | 0 | 3 | sub-022 | rsub-022_ses-1_acq-petra_run-02_PDw.nii | 0.6914 | 75.83 | 0.5284 |
| proposed | dicefocal | 0 | 4 | sub-022 | rsub-022_ses-2_acq-petra_run-02_PDw.nii | 0.7035 | 5.10 | 0.5426 |
| proposed | dicefocal | 0 | 5 | sub-022 | rsub-022_ses-3_acq-petra_run-01_PDw.nii | 0.7179 | 5.10 | 0.5599 |
| proposed | dicefocal | 0 | 6 | sub-022 | rsub-022_ses-4_acq-petra_run-01_PDw.nii | 0.3037 | 186.95 | 0.1790 |
| proposed | dicefocal | 0 | 7 | sub-030 | rsub-030_ses-2_acq-petra_run-01_PDw.nii | 0.7605 | 3.00 | 0.6135 |
| proposed | dicefocal | 0 | 8 | sub-030 | rsub-030_ses-3_acq-petra_run-02_PDw.nii | 0.7727 | 5.39 | 0.6296 |
| proposed | dicefocal | 0 | 9 | sub-040 | rsub-040_ses-2_acq-petra_run-01_PDw.nii | 0.7410 | 4.00 | 0.5886 |
| proposed | dicefocal | 0 | 10 | sub-044 | rsub-044_ses-2_acq-petra_run-02_PDw.nii | 0.2537 | 51.20 | 0.1453 |
| proposed | dicefocal | 1 | 0 | sub-001 | rsub-001_ses-3_acq-petra_run-01_PDw.nii | 0.6577 | 78.04 | 0.4899 |
| proposed | dicefocal | 1 | 1 | sub-001 | rsub-001_ses-4_acq-petra_run-02_PDw.nii | 0.6624 | 209.89 | 0.4952 |
| proposed | dicefocal | 1 | 2 | sub-002 | rsub-002_ses-1_acq-petra_run-01_PDw.nii | 0.5984 | 3.32 | 0.4270 |
| proposed | dicefocal | 1 | 3 | sub-002 | rsub-002_ses-1_acq-petra_run-02_PDw.nii | 0.5688 | 3.74 | 0.3975 |
| proposed | dicefocal | 1 | 4 | sub-002 | rsub-002_ses-2_acq-petra_run-01_PDw.nii | 0.5949 | 120.62 | 0.4234 |
| proposed | dicefocal | 1 | 5 | sub-006 | rsub-006_ses-2_acq-petra_run-01_PDw.nii | 0.5867 | 3.74 | 0.4152 |
| proposed | dicefocal | 1 | 6 | sub-015 | rsub-015_ses-1_acq-petra_run-02_PDw.nii | 0.7454 | 4.90 | 0.5941 |
| proposed | dicefocal | 1 | 7 | sub-021 | rsub-021_ses-3_acq-petra_run-01_PDw.nii | 0.7241 | 9.90 | 0.5675 |
| proposed | dicefocal | 1 | 8 | sub-024 | rsub-024_ses-1_acq-petra_run-01_PDw.nii | 0.7051 | 6.08 | 0.5445 |
| proposed | dicefocal | 1 | 9 | sub-038 | rsub-038_ses-4_acq-petra_run-01_PDw.nii | 0.6828 | 10.44 | 0.5184 |
| proposed | dicefocal | 1 | 10 | sub-052 | rsub-052_ses-4_acq-petra_run-01_PDw.nii | 0.7525 | 8.49 | 0.6032 |
| proposed | dicefocal | 2 | 0 | sub-009 | rsub-009_ses-1_acq-petra_run-01_PDw.nii | 0.6098 | 4.90 | 0.4386 |
| proposed | dicefocal | 2 | 1 | sub-018 | rsub-018_ses-4_acq-petra_run-01_PDw.nii | 0.7546 | 4.12 | 0.6059 |
| proposed | dicefocal | 2 | 2 | sub-023 | rsub-023_ses-2_acq-petra_run-02_PDw.nii | 0.7228 | 4.12 | 0.5660 |
| proposed | dicefocal | 2 | 3 | sub-029 | rsub-029_ses-1_acq-petra_run-01_PDw.nii | 0.5761 | 114.22 | 0.4046 |
| proposed | dicefocal | 2 | 4 | sub-029 | rsub-029_ses-3_acq-petra_run-02_PDw.nii | 0.6315 | 5.39 | 0.4615 |
| proposed | dicefocal | 2 | 5 | sub-036 | rsub-036_ses-4_acq-petra_run-01_PDw.nii | 0.7288 | 4.24 | 0.5733 |
| proposed | dicefocal | 2 | 6 | sub-043 | rsub-043_ses-1_acq-petra_run-02_PDw.nii | 0.7100 | 3.61 | 0.5504 |
| proposed | dicefocal | 2 | 7 | sub-054 | rsub-054_ses-2_acq-petra_run-01_PDw.nii | 0.0746 | 37.34 | 0.0387 |
| proposed | dicefocal | 2 | 8 | sub-054 | rsub-054_ses-2_acq-petra_run-02_PDw.nii | 0.0348 | 74.69 | 0.0177 |
| proposed | dicefocal | 2 | 9 | sub-054 | rsub-054_ses-3_acq-petra_run-02_PDw.nii | 0.5759 | 8.25 | 0.4044 |
| proposed | dicefocal | 2 | 10 | sub-073 | rsub-073_ses-2_acq-petra_run-01_PDw.nii | 0.7678 | 4.12 | 0.6231 |
| proposed | dicefocal | 3 | 0 | sub-004 | rsub-004_ses-1_acq-petra_run-02_PDw.nii | 0.6957 | 3.61 | 0.5333 |
| proposed | dicefocal | 3 | 1 | sub-016 | rsub-016_ses-2_acq-petra_run-01_PDw.nii | 0.7489 | 5.39 | 0.5986 |
| proposed | dicefocal | 3 | 2 | sub-016 | rsub-016_ses-2_acq-petra_run-02_PDw.nii | 0.7243 | 12.08 | 0.5677 |
| proposed | dicefocal | 3 | 3 | sub-016 | rsub-016_ses-3_acq-petra_run-01_PDw.nii | 0.7345 | 8.77 | 0.5804 |
| proposed | dicefocal | 3 | 4 | sub-017 | rsub-017_ses-3_acq-petra_run-01_PDw.nii | 0.6067 | 13.19 | 0.4355 |
| proposed | dicefocal | 3 | 5 | sub-017 | rsub-017_ses-4_acq-petra_run-01_PDw.nii | 0.7844 | 25.04 | 0.6453 |
| proposed | dicefocal | 3 | 6 | sub-032 | rsub-032_ses-3_acq-petra_run-01_PDw.nii | 0.7252 | 4.12 | 0.5688 |
| proposed | dicefocal | 3 | 7 | sub-034 | rsub-034_ses-4_acq-petra_run-02_PDw.nii | 0.6983 | 50.30 | 0.5365 |
| proposed | dicefocal | 3 | 8 | sub-037 | rsub-037_ses-4_acq-petra_run-01_PDw.nii | 0.5963 | 248.79 | 0.4248 |
| proposed | dicefocal | 3 | 9 | sub-058 | rsub-058_ses-3_acq-petra_run-01_PDw.nii | 0.2681 | 163.44 | 0.1548 |
| proposed | dicefocal | 3 | 10 | sub-062 | rsub-062_ses-4_acq-petra_run-01_PDw.nii | 0.1117 | 185.97 | 0.0592 |
| proposed | dicefocal | 4 | 0 | sub-003 | rsub-003_ses-3_acq-petra_run-02_PDw.nii | 0.5949 | 5.10 | 0.4234 |
| proposed | dicefocal | 4 | 1 | sub-005 | rsub-005_ses-4_acq-petra_run-02_PDw.nii | 0.7463 | 5.10 | 0.5953 |
| proposed | dicefocal | 4 | 2 | sub-007 | rsub-007_ses-3_acq-petra_run-02_PDw.nii | 0.0000 | 190.29 | 0.0000 |
| proposed | dicefocal | 4 | 3 | sub-007 | rsub-007_ses-4_acq-petra_run-01_PDw.nii | 0.5379 | 6.40 | 0.3679 |
| proposed | dicefocal | 4 | 4 | sub-010 | rsub-010_ses-3_acq-petra_run-01_PDw.nii | 0.7034 | 31.76 | 0.5424 |
| proposed | dicefocal | 4 | 5 | sub-010 | rsub-010_ses-4_acq-petra_run-01_PDw.nii | 0.5471 | 48.52 | 0.3766 |
| proposed | dicefocal | 4 | 6 | sub-013 | rsub-013_ses-1_acq-petra_run-01_PDw.nii | 0.6219 | 12.17 | 0.4512 |
| proposed | dicefocal | 4 | 7 | sub-013 | rsub-013_ses-2_acq-petra_run-02_PDw.nii | 0.7071 | 24.78 | 0.5469 |
| proposed | dicefocal | 4 | 8 | sub-020 | rsub-020_ses-2_acq-petra_run-02_PDw.nii | 0.7547 | 5.20 | 0.6061 |
| proposed | dicefocal | 4 | 9 | sub-066 | rsub-066_ses-1_acq-petra_run-02_PDw.nii | 0.7115 | 210.38 | 0.5522 |
| proposed | dicefocal | single | 0 | sub-006 | rsub-006_ses-2_acq-petra_run-01_PDw.nii | 0.6316 | 4.12 | 0.4616 |
| proposed | dicefocal | single | 1 | sub-009 | rsub-009_ses-1_acq-petra_run-01_PDw.nii | 0.5853 | 4.12 | 0.4137 |
| proposed | dicefocal | single | 2 | sub-010 | rsub-010_ses-3_acq-petra_run-01_PDw.nii | 0.6225 | 10.68 | 0.4519 |
| proposed | dicefocal | single | 3 | sub-010 | rsub-010_ses-4_acq-petra_run-01_PDw.nii | 0.0000 | 252.89 | 0.0000 |
| proposed | dicefocal | single | 4 | sub-015 | rsub-015_ses-1_acq-petra_run-02_PDw.nii | 0.7175 | 5.39 | 0.5594 |
| proposed | dicefocal | single | 5 | sub-032 | rsub-032_ses-3_acq-petra_run-01_PDw.nii | 0.5843 | 30.81 | 0.4128 |
| proposed | dicefocal | single | 6 | sub-036 | rsub-036_ses-4_acq-petra_run-01_PDw.nii | 0.7014 | 11.58 | 0.5402 |
| proposed | dicefocal | single | 7 | sub-040 | rsub-040_ses-2_acq-petra_run-01_PDw.nii | 0.6776 | 4.24 | 0.5125 |
| proposed | dicefocal | single | 0 | sub-010 | rsub-010_ses-3_acq-petra_run-01_PDw.nii | 0.7490 | 3.61 | 0.5988 |
| proposed | dicefocal | single | 1 | sub-010 | rsub-010_ses-4_acq-petra_run-01_PDw.nii | 0.6687 | 78.03 | 0.5023 |
| proposed | dicefocal | single | 2 | sub-012 | rsub-012_ses-4_acq-petra_run-02_PDw.nii | 0.6758 | 22.38 | 0.5103 |
| proposed | dicefocal | single | 3 | sub-023 | rsub-023_ses-2_acq-petra_run-02_PDw.nii | 0.7138 | 8.12 | 0.5550 |
| proposed | dicefocal | single | 4 | sub-044 | rsub-044_ses-2_acq-petra_run-02_PDw.nii | 0.7302 | 3.61 | 0.5750 |
| proposed | dicefocal | single | 5 | sub-052 | rsub-052_ses-4_acq-petra_run-01_PDw.nii | 0.7661 | 4.12 | 0.6208 |
| proposed | dicefocal | single | 6 | sub-4143 | rsub-4143_ses-1_acq-petra_run-01_PDw.nii | 0.0363 | 172.29 | 0.0185 |
| proposed | dicefocal | single | 7 | sub-6103 | rsub-6103_ses-1_acq-petra_run-02_PDw.nii | 0.5485 | 55.44 | 0.3778 |
