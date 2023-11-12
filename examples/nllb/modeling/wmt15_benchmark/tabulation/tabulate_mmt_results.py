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
# model_dir = sys.argv[1]
model_dir = "/large_experiments/moe/shru/wmt30/many_to_en/dense.fp16.mu100000.uf1.entsrc.SPL_temperature.tmp5.adam.beta0.9_0.98.archtransformer_wmt_en_de.dpreln.epreln.initlr1e-07.warmup8000.lr0.0005.clip0.0.drop0.0.wd0.0.ls0.1.maxtok3584.seed2.max_pos512.no_c10d.det.ves100000000.ngpu32"
model_dir = "/large_experiments/moe/shru/wmt30/many_to_en/dense.fp16.mu100000.uf1.entsrc.SPL_temperature.tmp5.adam.beta0.9_0.98.archtransformer_wmt_en_de.dpreln.epreln.initlr1e-07.warmup8000.lr0.002.clip0.0.drop0.0.wd0.0.ls0.1.maxtok3584.seed2.max_pos512.no_c10d.det.ves100000000.ngpu32"
model_dir = "/large_experiments/moe/shru/wmt30/many_to_en/dense.fp16.mu100000.uf1.entsrc.SPL_temperature.tmp5.adam.beta0.9_0.98.archtransformer_wmt_en_de.dpreln.epreln.initlr1e-07.warmup8000.lr0.002.clip0.0.drop0.0.wd0.0.ls0.1.maxtok7168.seed2.max_pos512.no_c10d.det.ves100000000.ngpu32"
UPDATES_LST = [10000 * (i + 1) for i in range(10)]
# get valid PPLs per language per #updates
valid_ppls = {}
for lang in LANGS:
    src = lang
    tgt = "en"
    valid_prefix = f"valid_main:{src}-{tgt}"
    valid_ppls[lang] = {}
    with open(model_dir + "/train.log", "r") as f:
        for line in f.readlines():
            # valid_main:hi-en_ppl
            if "_ppl" in line and valid_prefix in line:
                parts = line.split(" | ")
                json_data = json.loads(parts[-1])
                num_updates = int(json_data[valid_prefix + "_num_updates"])
                ppl = float(json_data[valid_prefix + "_ppl"])
                valid_ppls[lang][num_updates] = ppl

print(valid_ppls)

# get valid, test BLEU per language and per # updates


def get_bleus(type="valid"):
    bleus = {}
    for lang in LANGS:
        src = lang
        tgt = "en"
        bleus[lang] = {}
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
                bleus[lang][updates] = float(output)
            except:
                bleus[lang][updates] = -1
    return bleus


valid_bleus = get_bleus("valid")
test_bleus = get_bleus("test")


def print_bleus(type):
    print(
        f"{type}\t"
        + "\t".join([str(u) for u in UPDATES_LST])
        + "\tBEST BLEU\tBEST_CHECKPOINT"
    )

    bleus = valid_bleus if type == "valid" else test_bleus
    for lang in LANGS:
        src = lang
        tgt = "en"
        out_line = f"{src}-{tgt}\t" + "\t".join(
            [f"{bleus[lang][u]}" for u in UPDATES_LST]
        )
        best_updates = max(bleus[lang], key=bleus[lang].get)
        out_line += f"\t{bleus[lang][best_updates]}\t{best_updates}"
        # print(out_line)
        if type == "valid":
            out_line = f"{src}-{tgt}-valid-ppl\t" + "\t".join(
                [f"{valid_ppls[lang][u]}" for u in UPDATES_LST]
            )
            best_updates = min(valid_ppls[lang], key=valid_ppls[lang].get)
            out_line += f"\t{valid_ppls[lang][best_updates]}\t{best_updates}"
            print(out_line)


# print_bleus(type="test")
print_bleus(type="valid")
