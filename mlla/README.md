# Ablation Study

<pre>
logs
├── 1_mlla_nano
│   ├── <a style="text-decoration:none" href="./logs/1_mlla_nano/01_baseline.txt">01_baseline.txt</a>
│   ├── <a style="text-decoration:none" href="./logs/1_mlla_nano/02_recconv_5x5_conv_trans.txt">02_recconv_5x5_conv_trans.txt</a>
│   ├── <a style="text-decoration:none" href="./logs/1_mlla_nano/03_recconv_5x5_nearest_interp.txt">03_recconv_5x5_nearest_interp.txt</a>
│   ├── <a style="text-decoration:none" href="./logs/1_mlla_nano/04_recattn_nearest_interp.txt">04_recattn_nearest_interp.txt</a>
│   └── <a style="text-decoration:none" href="./logs/1_mlla_nano/05_recattn_nearest_interp_simplify.txt">05_recattn_nearest_interp_simplify.txt</a>
└── 2_mlla_mini
    ├── <a style="text-decoration:none" href="./logs/2_mlla_mini/01_baseline.txt">01_baseline.txt</a>
    ├── <a style="text-decoration:none" href="./logs/2_mlla_mini/02_recconv_5x5_conv_trans.txt">02_recconv_5x5_conv_trans.txt</a>
    ├── <a style="text-decoration:none" href="./logs/2_mlla_mini/03_recconv_5x5_nearest_interp.txt">03_recconv_5x5_nearest_interp.txt</a>
    ├── <a style="text-decoration:none" href="./logs/2_mlla_mini/04_recattn_nearest_interp.txt">04_recattn_nearest_interp.txt</a>
    └── <a style="text-decoration:none" href="./logs/2_mlla_mini/05_recattn_nearest_interp_simplify.txt">05_recattn_nearest_interp_simplify.txt</a>
</pre>

```bash
# this script is used to validate the ablation results
fd txt logs -x sh -c 'printf "%.2f %s\n" "$(rg -N -I -U -o "EPOCH.*\n.*Acc@1 (\d+\.\d+)" -r "\$1" {} | sort -n | tail -1)" "{}"' | sort -k2
```

<details>
  <summary>
  <span>output</span>
  </summary>

```
76.26 logs/1_mlla_nano/01_baseline.txt
77.09 logs/1_mlla_nano/02_recconv_5x5_conv_trans.txt
77.14 logs/1_mlla_nano/03_recconv_5x5_nearest_interp.txt
76.53 logs/1_mlla_nano/04_recattn_nearest_interp.txt
77.28 logs/1_mlla_nano/05_recattn_nearest_interp_simplify.txt
82.27 logs/2_mlla_mini/01_baseline.txt
82.06 logs/2_mlla_mini/02_recconv_5x5_conv_trans.txt
81.94 logs/2_mlla_mini/03_recconv_5x5_nearest_interp.txt
82.08 logs/2_mlla_mini/04_recattn_nearest_interp.txt
82.16 logs/2_mlla_mini/05_recattn_nearest_interp_simplify.txt
```
</details>