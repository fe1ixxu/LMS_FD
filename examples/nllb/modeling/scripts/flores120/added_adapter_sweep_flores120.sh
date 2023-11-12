# bash examples/nllb/modeling/scripts/flores120/added_adapter_sweep_flores120.sh moe/dense
partition=learnaccel
num_trials=1
num_nodes=1
num_gpus_per_node=8
mem="0"
max_update=60000
max_tokens=2048
update_freq=2
max_time_mins=4319 # 3-day limit for learnaccel
arch=added_adapter_transformer
validate_interval_updates=5000
seed=3


script_module="examples.nllb.modeling.sweep.sweep_mmt"

# fairseq="/shared/home/shru/projects/fairseq-py"
# SET TO LOCAL FAIRSEQ REPO DIRECTORY
fairseq=/private/home/jcross/fairseq-py
cd $fairseq
langs_file="$fairseq/examples/nllb/modeling/scripts/flores120/langs.txt"

eval_cap=1.0

data_prefix="/large_experiments/nllb/mmt/multilingual_bin/flores125.en_xx_en.v3.3/data_bin"

num_shards=32
data_dir=""
for (( shard_num=0; shard_num<$num_shards; shard_num++ ))
do
  shard_id=$(printf "%03d" $shard_num)
  data_dir=${data_dir}:${data_prefix}/shard${shard_id}/
done
data_dir=${data_dir:1}
echo $data_dir

chk_prefix="/large_experiments/nllb/mmt/jcross/flores120/added_adapters"

if [ "$1" == "moe" ] ; then
    checkpoint_dir="${chk_prefix}/final_human_moe_2048/"
    moe_param="--moe-base-model"
else
    checkpoint_dir="${chk_prefix}/final_human_dense_2048.lr001/"
    moe_param=""
fi

encoder_langtok="tgt"
warmup=4000
ddp_backend="no_c10d"

# dense
# base_model=/large_experiments/nllb/mmt/h2_21_models/flores125_v3.3/en_to_many_to_en/v3.3_dense.mfp16.mu100000.uf4.lss.enttgt.tmp1.5.shem.NBF.warmup8000.lr0.004.drop0.0.maxtok2560.seed2.valevery200000.max_pos512.adam16bit.fully_sharded.det.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H16.ATTDRP0.1.RELDRP0.0.ngpu128/checkpoint_10_100000_consolidated.pt
# MoE
# base_model=/large_experiments/nllb/mmt/h2_21_models/flores125_v3.3/64.mfp16.mu100000.uf3.lss.tmp1.shem.NBF.warmup8000.lr0.004.drop0.0.maxtok5120.seed2.max_pos512.adam16bit.moe_w0.01.all.fully_sharded.enttgt.det.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H16.ATTDRP0.1.RELDRP0.0.ngpu64/checkpoint_10_85000-rank-0.pt

base_model=/large_experiments/nllb/mmt/h2_21_models/flores125_v3.3/en_to_many_to_en/v3.3_dense_hrft004.mfp16.mu100000.uf4.lss.enttgt.tmp1.0.shem.NBF.warmup8000.lr0.004.drop0.0.maxtok2560.seed2.valevery200000.max_pos512.adam16bit.fully_sharded.det.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H16.ATTDRP0.1.RELDRP0.0.ngpu128/checkpoint_15_100000_consolidated.pt

# evaluation subsample
# for langpair in eng-ara eng-asm eng-ben eng-ceb eng-hau eng-hin eng-ibo eng-ind eng-isl eng-jpn eng-kat eng-kor eng-lav eng-mon eng-oci eng-rus eng-snd eng-tam eng-tha eng-tur eng-vie eng-xho eng-zho_Hans eng-zul ; do
# for langpair in ara-eng asm-eng ben-eng ceb-eng hau-eng hin-eng ibo-eng ind-eng isl-eng jpn-eng kat-eng kor-eng lav-eng mon-eng oci-eng rus-eng snd-eng tam-eng tha-eng tur-eng vie-eng xho-eng zho_Hans-eng zul-eng ; do

# human-evaluation directions
for langpair in amh-eng ara-eng asm-eng azj-eng bos-eng bul-eng ceb-eng ces-eng deu-eng eng-amh eng-ara eng-asm eng-azj eng-bos eng-bul eng-ceb eng-ces eng-deu eng-fin eng-gla eng-gle eng-hau eng-hin eng-hye eng-isl eng-jpn eng-kat eng-mon eng-msa eng-orm eng-ron eng-snd eng-sqi eng-ssw eng-swh eng-tam eng-tha eng-urd eng-wol eng-zho_Hans eng-zul fin-eng gla-eng gle-eng hau-eng hin-eng hye-eng isl-eng jpn-eng kat-eng mon-eng msa-eng orm-eng ron-eng snd-eng sqi-eng ssw-eng swh-eng tam-eng tha-eng urd-eng wol-eng zho_Hans-eng zul-eng ; do
# remaining directions
# for langpair in afr-eng ast-eng aym-eng bel-eng ben-eng cat-eng ckb-eng cym-eng dan-eng dyu-eng ell-eng eng-afr eng-ast eng-aym eng-bel eng-ben eng-cat eng-ckb eng-cym eng-dan eng-dyu eng-ell eng-est eng-fas eng-fra eng-ful eng-glg eng-guj eng-hat eng-heb eng-hrv eng-hun eng-ibo eng-ilo eng-ind eng-ita eng-jav eng-kac eng-kam eng-kan eng-kaz eng-kea eng-khm eng-kir eng-kmb eng-kon eng-kor eng-kur eng-lao eng-lav eng-lin eng-lit eng-ltz eng-lug eng-luo eng-mal eng-mar eng-mkd eng-mlg eng-mlt eng-mri eng-mya eng-nld eng-nob eng-npi eng-nso eng-nya eng-oci eng-ory eng-pan eng-pol eng-por eng-pus eng-que eng-rus eng-sin eng-slk eng-slv eng-sna eng-som eng-spa eng-srp eng-sun eng-swe eng-tel eng-tgk eng-tgl eng-tir eng-tsn eng-tur eng-ukr eng-umb eng-uzb eng-vie eng-xho eng-yid eng-yor eng-yue est-eng fas-eng fra-eng ful-eng glg-eng guj-eng hat-eng heb-eng hrv-eng hun-eng ibo-eng ilo-eng ind-eng ita-eng jav-eng kac-eng kam-eng kan-eng kaz-eng kea-eng khm-eng kir-eng kmb-eng kon-eng kor-eng kur-eng lao-eng lav-eng lin-eng lit-eng ltz-eng lug-eng luo-eng mal-eng mar-eng mkd-eng mlg-eng mlt-eng mri-eng mya-eng nld-eng nob-eng npi-eng nso-eng nya-eng oci-eng ory-eng pan-eng pol-eng por-eng pus-eng que-eng rus-eng sin-eng slk-eng slv-eng sna-eng som-eng spa-eng srp-eng sun-eng swe-eng tel-eng tgk-eng tgl-eng tir-eng tsn-eng tur-eng ukr-eng umb-eng uzb-eng vie-eng xho-eng yid-eng yor-eng yue-eng ; do
    prefix=$langpair

    for lr in 0.001 ; do # 0.0005 0.001 0.003 ; do
        python -m ${script_module} -d ${data_dir} -p "$prefix" \
            --checkpoints-dir ${checkpoint_dir} --partition ${partition} --constraint volta32gb \
            -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
            --time "$max_time_mins" \
            --mem ${mem} \
            --sampling-method temperature --sampling-temperature 1 \
            --decoder-langtok --encoder-langtok ${encoder_langtok} \
            --langs $langs_file \
            --lang-pairs $langpair \
            ${moe_param} \
            --moe-eval-cap ${eval_cap} \
            --ddp-backend $ddp_backend \
            --max-update $max_update \
            --max-tokens $max_tokens \
            --update-freq $update_freq \
            --warmup-updates ${warmup} \
            --lr $lr \
            --opt adam16bit \
            --share-all-embeddings \
            --save-interval-updates 5000 \
            --tensorboard-logdir ${chk_prefix}/tb/ \
            --arch "$arch" \
            --dropout 0.0 \
            --validate-interval-updates "${validate_interval_updates}" \
            --seed "$seed" \
            --snapshot-code \
            --use-local-shard-size \
            --base-model $base_model \
            --adapter-hidden-dim 2048
    done
    sleep 7
done
