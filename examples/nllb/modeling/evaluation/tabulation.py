import glob
import json
import os
import pdb
import subprocess
import sys

# import wandb

LANGS = [
    "afr",
    "amh",
    "ara",
    "asm",
    "ast",
    "azj",
    "bel",
    "ben",
    "bos",
    "bul",
    "cat",
    "ceb",
    "ces",
    "cym",
    "dan",
    "deu",
    "ell",
    "est",
    "fas",
    "fin",
    "fra",
    "ful",
    "gle",
    "glg",
    "guj",
    "hau",
    "heb",
    "hin",
    "hrv",
    "hun",
    "hye",
    "ibo",
    "ind",
    "isl",
    "ita",
    "jav",
    "jpn",
    "kam",
    "kan",
    "kat",
    "kaz",
    "kea",
    "khm",
    "kir",
    "kor",
    "kur",
    "lao",
    "lav",
    "lin",
    "lit",
    "ltz",
    "lug",
    "luo",
    "mal",
    "mar",
    "mkd",
    "mlt",
    "mon",
    "mri",
    "msa",
    "mya",
    "nld",
    "nob",
    "npi",
    "nso",
    "nya",
    "oci",
    "orm",
    "ory",
    "pan",
    "pol",
    "por",
    "pus",
    "ron",
    "rus",
    "slk",
    "slv",
    "sna",
    "snd",
    "som",
    "spa",
    "srp",
    "swe",
    "swh",
    "tam",
    "tel",
    "tgk",
    "tgl",
    "tha",
    "tur",
    "ukr",
    "umb",
    "urd",
    "vie",
    "wol",
    "xho",
    "yor",
    "zho",
    "zul",
]

MODEL_DIR = ""

# TODO : Add W&B support
# run = wandb.init(project='evaluations', entity='vedanuj')
# text_table = wandb.Table(columns=["lang", "spmBLEU"])


def get_bleus(type="valid"):
    bleus = {}
    for lang in LANGS:
        tgt = "eng"
        src = lang
        bleus[lang] = {}
        try:
            fname = list(
                glob.glob(
                    os.path.join(
                        MODEL_DIR, f"{src}-{tgt}_checkpoint_last_{type}", "bleu.results"
                    )
                )
            )[0]
            command = f"grep BLEU {fname} | cut -f3 -d' '"
            output = subprocess.check_output(command, shell=True).decode()
            bleus[lang] = float(output)
        except:
            bleus[lang] = -1
    return bleus


test_bleus = get_bleus("valid")


def print_bleus(type):
    bleus = test_bleus
    for lang in LANGS:
        tgt = "eng"
        src = lang
        out_line = "\t".join([f"{bleus[lang]}"])
        # text_table.add_data(f"{lang}-eng", bleus[lang])
        print(out_line)


print_bleus(type="valid")

# run.log({"training_samples" : text_table})
