/private/home/namangoyal/src/sentencepiece/build/src/spm_train \
--input "/private/home/shru/wmt_datasets/raw_v2/spm_samples.txt" \
--model_prefix "/private/home/shru/wmt_datasets/raw_v2/spm_64000_50M" \
--vocab_size=64000 \
--character_coverage=0.99995 \
--model_type=bpe \
--shuffle_input_sentence=true \
--input_sentence_size=500000000 \
--num_threads 72;
