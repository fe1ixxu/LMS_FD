# read
import glob
import os

langs = set()

flores_to_cc100_langs = {}
flores_to_iso_langs = {}
with open("lang_codes_mapping.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        if i == 0:
            continue
        else:
            parts = line.strip().split("\t")
            flores_to_cc100_langs[parts[1]] = parts[2]
            flores_to_iso_langs[parts[1]] = parts[0]
print(flores_to_cc100_langs)
print(flores_to_iso_langs)

langs = flores_to_cc100_langs.keys()

recent_mining_langs = set()
files = glob.glob("/large_experiments/mmt/mining/data/mini-mine1/bitexts/*")
holger_langs = set()
for file in files:
    if len(os.path.basename(file).split(".")) > 2:
        holger_langs.add(os.path.basename(file).split(".")[0].split("-")[1])
        holger_langs.add(os.path.basename(file).split(".")[0].split("-")[0])


files = glob.glob("/checkpoint/angelafan/data/cc100_multilingual/raw_pre_spm/*")
directions = set()
old_m2m100_langs = set()
for file in files:
    directions.add(file.split(".")[1].strip())
    old_m2m100_langs.add(file.split(".")[1].strip().split("-")[0])
    old_m2m100_langs.add(file.split(".")[1].strip().split("-")[1])

# print(directions)
# print(len(directions))

en_x = {}
in_old_m2m100 = set()
for lang in langs:
    if not f"en-{lang}" in directions and not f"{lang}-en" in directions:
        # if not lang in old_m2m100_langs:
        cc100_lang = (
            flores_to_cc100_langs[lang] if lang in flores_to_cc100_langs else lang
        )
        if (
            not f"en-{cc100_lang}" in directions
            and not f"{cc100_lang}-en" in directions
        ):
            # if not cc100_lang in old_m2m100_langs:
            print(lang)
        else:
            in_old_m2m100.add(lang)
    else:
        in_old_m2m100.add(lang)

print("===")
in_holger_mined = set()
for lang in langs:
    lang2 = flores_to_iso_langs[lang] if lang in flores_to_iso_langs else lang
    if lang2 in holger_langs:
        in_holger_mined.add(lang)
worse_in_new_mined = {"lug", "luo", "nso", "nya", "orm", "sna"}

print("== only in old m2m100 ==")
for lang in langs:
    if lang in in_old_m2m100 and not lang in in_holger_mined:
        print(lang)

print("== in both old and mined ==")
for lang in langs:
    if lang in in_old_m2m100 and lang in in_holger_mined:
        print(lang, flores_to_iso_langs[lang])

print("== in only mined ==")
for lang in langs:
    if not lang in in_old_m2m100 and lang in in_holger_mined:
        print(lang, flores_to_iso_langs[lang])

print("== in neither ==")
for lang in langs:
    if not lang in in_old_m2m100 and not lang in in_holger_mined:
        print(lang, flores_to_iso_langs[lang])


print("== use from old m2m100 ==")
num_old = 0
for lang in langs:
    if lang in in_old_m2m100 and not lang in in_holger_mined:
        print(lang)
        num_old += 1
    elif (
        lang in in_old_m2m100
        and lang in in_holger_mined
        and flores_to_iso_langs[lang] in worse_in_new_mined
    ):
        print(lang)
        num_old += 1
    if flores_to_iso_langs[lang] in worse_in_new_mined and not lang in in_old_m2m100:
        print(f"worse holger mining bleu, but not in old data I have={lang}")
print(f"count={num_old}")

print("== use from new mined ==")
num_new = 0
for lang in langs:
    if lang in in_holger_mined and not flores_to_iso_langs[lang] in worse_in_new_mined:
        print(flores_to_iso_langs[lang])
        num_new += 1
print(f"count={num_new}")
