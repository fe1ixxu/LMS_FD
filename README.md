This is the repo for our EMNLP 2023 paper: [Condensing Multilingual Knowledge with Lightweight Language-Specific Modules](https://browse.arxiv.org/pdf/2305.13993.pdf)
```
@article{xu2023condensing,
  title={Condensing Multilingual Knowledge with Lightweight Language-Specific Modules},
  author={Xu, Haoran and Tan, Weiting and Li, Shuyue Stella and Chen, Yunmo and Van Durme, Benjamin and Koehn, Philipp and Murray, Kenton},
  journal={arXiv preprint arXiv:2305.13993},
  year={2023}
}
```

## Building VirtualEnvironments:
```
conda create -n lms python=3.8
conda activate lms
bash install.sh
```

## Download the Preprocessed Data
One can download our preprocessed OPUS-100 dataset by running:
```
gdown 1owwSARAf95EpiWz7PeTNu-kbQ3cuMCh4
unzip opus-15.zip
```

and download preprocessed OPUS-15 (the 15-language ablation study dataset selected from OPUS-100) by running:
```
gdown 1X-Zj2wcCdR2zpEYA-_CcaBGoEvF_6jNS
unzip opus-100.zip
```
## Training & Evaluation
To train the naive MMT model on OPUS-100:
```
bash ./runs/train_opus_100_baseline.sh
```
and OPUS-15:
```
bash ./runs/train_opus_15_baseline.sh
```

To reproduce LMS or LMS+FD results on OPUS-100 data:
```
bash ./runs/train_opus_100.sh ${LMS_RANK} ${LMS_FREQ} ${LMS_TYPE}
```

The three variants are defined as follows:

- `${LMS_RANK}`: Represents the hidden dimension size of the vertical and flatten matrix. The default value is `4`.
- `${LMS_FREQ}`: Determines the frequency of implementing LMS in transformer layers. For example, `1` means one LMS layer per transformer layer, while `2` means one LMS layer per two transformer layers. The default value is `1`.
- `${LMS_TYPE}`: Specifies the type of LMS, including `pair` (pair-wise LMS), lang (language-wise LMS), `pair_fd` (pair-wise LMS with fused distillation), and `lang_fd` (language-wise LMS with fused distillation). The default value is `pair`.

**Evaluation will be automatically conducted after training is finished.**

Similarly, to reproduce the results of OPUS-15:
```
bash ./runs/train_opus_15.sh ${LMS_RANK} ${LMS_FREQ} ${LMS_TYPE}
```



