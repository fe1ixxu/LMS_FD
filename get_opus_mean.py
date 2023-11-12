import argparse

HIGH = list({'nl', 'ca', 'fi', 'mk', 'da', 'cs', 'bg', 'ro', 'is', 'th', 'he', 'uk', 'lv', 'pl', 'pt', 'hu', 'de', 'lt',
                  'si', 'ms', 'sv', 'tr', 'ko', 'sq', 'el', 'fa', 'es', 'zh', 'bs', 'ar', 'eu', 'fr', 'bn', 'it', 'sk', 'sr',
                  'et', 'vi', 'mt', 'no', 'sl', 'id', 'ja', 'ru', 'hr'})
LOW = list({'ga', 'af', 'tg', 'gu', 'km', 'sh', 'hi', 'rw', 'nb', 'wa', 'uz', 'ka', 'ml', 'ur', 'gl', 'br', 'cy', 'ku',
                 'ne', 'pa', 'mg', 'as', 'eo', 'xh', 'nn', 'ta', 'az', 'tt'})
VERY_LOW= list({'fy', 'mr', 'tk', 'kn', 'li', 'yi', 'my', 'zu', 'ug', 'or', 'se', 'am', 'oc', 'ig', 'ha', 'ky', 'te', 'be',
                 'kk', 'gd', 'ps'})

ALL = HIGH+LOW+VERY_LOW


def get_m15_mean(args):
    all_bleu_scores = []
    high_bleu_scores = []
    low_bleu_scores = []
    very_low_bleu_scores = []

    
    for lang in ALL:
        try:
            if args.engtgt:
                src, tgt = lang, "en"
                bleu_file = f"/predict.{tgt}-{src}.{tgt}.bleu"
            else:
                src, tgt = "en", lang
                bleu_file = f"/predict.{src}-{tgt}.{tgt}.bleu"
            ## BLEU
            with open(args.input + bleu_file) as f:
                l = f.readlines()
            bleu = float(l[0].split("=")[1].split(" ")[1])

            all_bleu_scores.append(bleu)
            if lang in LOW:
                low_bleu_scores.append(bleu)
            if lang in VERY_LOW:
                very_low_bleu_scores.append(bleu)
            elif lang in HIGH:
                high_bleu_scores.append(bleu)
        except:
            print("No results for the language {}".format(lang))
    def get_mean(scores):
        return round(sum(scores)/len(scores) ,2)
    print(f"Avg of all BLEU is {get_mean(all_bleu_scores)}")
    print(f"Avg of high resource BLEU is {get_mean(high_bleu_scores)}")
    print(f"Avg of low resource BLEU is {get_mean(low_bleu_scores)}")
    print(f"Avg of very low resource is {get_mean(very_low_bleu_scores)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input folder')
    parser.add_argument('--engtgt', type=int, default=0, help='eng is tgt')
    args = parser.parse_args()
    get_m15_mean(args)

