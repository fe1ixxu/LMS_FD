import json
import os

old_m2m100_data = "/checkpoint/angelafan/data/cc100_multilingual/raw_pre_spm"
holger_mined_data = "/checkpoint/vishrav/mmt/bitexts_v1"

cc100_langs_to_iso = {}
iso_to_cc100_langs = {}
flores_to_iso = {}
with open("lang_codes_mapping.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        if i == 0:
            continue
        else:
            parts = line.strip().split("\t")
            cc100_langs_to_iso[parts[2]] = parts[0]
            iso_to_cc100_langs[parts[0]] = parts[2]
            flores_to_iso[parts[1]] = parts[0]
print(cc100_langs_to_iso)


def get_lang_set(fpath):
    lang_set = set()
    with open(fpath, "r") as f:
        for line in f.readlines():
            lang_set.add(line.strip())
    return lang_set


old_m2m100_langs = get_lang_set("old_m2m100_langs.txt")
print(old_m2m100_langs)

en_paths = dict()
x_paths = dict()
for lang in old_m2m100_langs:
    iso_lang = cc100_langs_to_iso[lang] if lang in cc100_langs_to_iso else lang
    from_en_path1 = os.path.join(old_m2m100_data, f"train.en-{lang}.en")
    from_en_path2 = os.path.join(old_m2m100_data, f"train.en-{lang}.{lang}")
    from_x_path1 = os.path.join(old_m2m100_data, f"train.{lang}-en.en")
    from_x_path2 = os.path.join(old_m2m100_data, f"train.{lang}-en.{lang}")
    if (
        os.path.exists(from_en_path1)
        and os.path.exists(from_en_path2)
        and os.path.exists(from_x_path1)
        and os.path.exists(from_x_path2)
    ):
        print(lang)
    if os.path.exists(from_en_path1) and os.path.exists(from_en_path2):
        en_paths[iso_lang] = from_en_path1
        x_paths[iso_lang] = from_en_path2
    elif os.path.exists(from_x_path1) and os.path.exists(from_x_path2):
        en_paths[iso_lang] = from_x_path1
        x_paths[iso_lang] = from_x_path2
    else:
        print(lang)

holger_mining_langs = get_lang_set("holger_mining_langs.txt")
print(holger_mining_langs)

for lang in holger_mining_langs:
    from_en_path1 = os.path.join(holger_mined_data, f"train.eng-{lang}.eng")
    from_en_path2 = os.path.join(holger_mined_data, f"train.eng-{lang}.{lang}")
    from_x_path1 = os.path.join(holger_mined_data, f"train.{lang}-eng.eng")
    from_x_path2 = os.path.join(holger_mined_data, f"train.{lang}-eng.{lang}")
    if os.path.exists(from_en_path1) and os.path.exists(from_en_path2):
        en_paths[lang] = from_en_path1
        x_paths[lang] = from_en_path2
    elif os.path.exists(from_x_path1) and os.path.exists(from_x_path2):
        en_paths[lang] = from_x_path1
        x_paths[lang] = from_x_path2
    else:
        print(lang)


opus_mining_langs = get_lang_set("opus_mining_langs.txt")

for lang in opus_mining_langs:
    from_en_path1 = os.path.join(holger_mined_data, f"train.en-{lang}.eng")
    from_en_path2 = os.path.join(holger_mined_data, f"train.en-{lang}.{lang}")
    from_x_path1 = os.path.join(holger_mined_data, f"train.{lang}-en.eng")
    from_x_path2 = os.path.join(holger_mined_data, f"train.{lang}-en.{lang}")
    iso_lang = flores_to_iso[lang]
    if os.path.exists(from_en_path1) and os.path.exists(from_en_path2):
        en_paths[iso_lang] = from_en_path1
        x_paths[iso_lang] = from_en_path2
    elif os.path.exists(from_x_path1) and os.path.exists(from_x_path2):
        en_paths[iso_lang] = from_x_path1
        x_paths[iso_lang] = from_x_path2
    else:
        print(lang)
ofile = open("lang_file_mapping.txt", "w")
# print("{", file=ofile)
# print(len(en_paths))
# line_format = """   "{l1}-{l2}": {{
#         "{data_source}": {{
#             "src": "{src_path}",
#             "tgt": "{tgt_path}",
#         }},
#     }},"""
output_dict = dict()
for lang in en_paths.keys():
    if lang in holger_mining_langs:
        data_source = "mining_jul21"
    else:
        data_source = "m2m100"
    for type in ["to_en", "from_en"]:
        if type == "to_en":
            src = lang
            tgt = "eng"
            src_path = x_paths[lang]
            tgt_path = en_paths[lang]
        else:
            src = "eng"
            tgt = lang
            src_path = en_paths[lang]
            tgt_path = x_paths[lang]
        output_dict[f"{src}-{tgt}"] = dict()
        output_dict[f"{src}-{tgt}"][f"{data_source}"] = dict()
        output_dict[f"{src}-{tgt}"][f"{data_source}"]["src"] = src_path
        output_dict[f"{src}-{tgt}"][f"{data_source}"]["tgt"] = tgt_path
        # print(line_format.format(l1=src, l2=tgt, data_source=data_source, src_path=src_path, tgt_path=tgt_path), file=ofile)

# print("}", file=ofile)
# ofile.close()
json.dump(output_dict, ofile, indent=2)
ofile.close()
