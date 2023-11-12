mkdir ~/wmt_datasets/raw_v2/many_to_en_bin
mkdir ~/wmt_datasets/raw_v2/en_to_many_bin
for lang in cs de es  et  'fi'  fr  gu  hi  kk  lt  lv  ro  ru  tr  zh ; do
    ln -s ~/wmt_datasets/raw_v2/binarized/$lang-en/* ~/wmt_datasets/raw_v2/many_to_en_bin/
done

for lang in cs de es  et  'fi'  fr  gu  hi  kk  lt  lv  ro  ru  tr  zh ; do
    ln -s ~/wmt_datasets/raw_v2/binarized/en-$lang/* ~/wmt_datasets/raw_v2/en_to_many_bin/
done
