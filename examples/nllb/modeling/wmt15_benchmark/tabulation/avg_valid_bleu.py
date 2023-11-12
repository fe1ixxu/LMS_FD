import glob
import json
import os
import pdb
import subprocess
import sys

LANGS = [
    "cs",
    "de",
    "es",
    "et",
    "fi",
    "fr",
    "gu",
    "hi",
    "kk",
    "lt",
    "lv",
    "ro",
    "ru",
    "tr",
    "zh",
]
# UPDATES_LST = [10000 * (i + 1) for i in range(10)]
UPDATES_LST = [100000]

# get valid, test BLEU per language and per # updates


def get_avg_bleu(model_dir, type="valid", updates="100000"):
    total_bleu = 0.0
    for lang in LANGS:
        src = lang
        tgt = "en"
        for updates in UPDATES_LST:
            try:
                fname = list(
                    glob.glob(
                        os.path.join(
                            model_dir,
                            "gen_output",
                            f"{src}-{tgt}_checkpoint_*_{updates}_{type}",
                            "bleu.results",
                        )
                    )
                )[0]
                command = f"grep BLEU {fname} | cut -f3 -d' '"
                output = subprocess.check_output(command, shell=True).decode()
                print(f"lang={lang} bleu={output}")
                total_bleu += float(output)
            except:
                print(fname)
                total_bleu += -1000
    return round(total_bleu / len(LANGS), 2)


for warmup in [8000]:
    for mt in [7168]:
        for lr in [0.002]:
            model_dir = f"/large_experiments/moe/shru/wmt30/many_to_en/dense.fp16.mu100000.uf1.entsrc.SPL_temperature.tmp5.adam.beta0.9_0.98.archtransformer_wmt_en_de.dpreln.epreln.initlr1e-07.warmup{warmup}.lr{lr}.clip0.0.drop0.0.wd0.0.ls0.1.maxtok{mt}.seed2.max_pos512.no_c10d.det.ves100000000.ngpu32"
            print(model_dir)
            avg_valid_bleu = get_avg_bleu(model_dir)
            print(f"warmup={warmup}, mt={mt}, lr={lr}, avg_valid_bleu={avg_valid_bleu}")
