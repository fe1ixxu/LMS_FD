# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import asyncio
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import tqdm
from submitit import AutoExecutor
from torch.nn.functional import cosine_similarity

from examples.nllb.modeling.utils import awaitable_job
from fairseq import checkpoint_utils

sns.set_theme(style="whitegrid", palette="pastel")

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("grad_similarity")


def zero_grad(model):
    for p in model.parameters():
        p.grad = None


def grad2vec(model, modules="all"):
    if modules == "all":
        return torch.cat([p.grad.data.view(-1) for p in model.parameters()])
    if modules == "encoder":
        return torch.cat([p.grad.data.view(-1) for p in model.encoder.parameters()])
    if modules == "encoder\emb":
        return torch.cat(
            [
                p.grad.data.view(-1)
                for p in model.encoder.parameters()
                if not p.size()[0] == 64006
            ]
        )
    if modules == "decoder":
        return torch.cat([p.grad.data.view(-1) for p in model.decoder.parameters()])
    if modules == "decoder\emb":
        return torch.cat(
            [
                p.grad.data.view(-1)
                for p in model.decoder.parameters()
                if not p.size()[0] == 64006
            ]
        )


def compute_gradients(langdir, task, model, criterion, split="valid_main"):
    model.train()
    zero_grad(model)
    itr = task.get_batch_iterator(
        dataset=task.datasets[split + ":" + langdir],
        max_tokens=6000,
        ignore_invalid_inputs=True,
    ).next_epoch_itr(shuffle=False)
    full_split_loss = 0.0
    count = 0
    for sample in itr:
        # sample = move_to_cuda(sample)
        loss, sample_size, logging_output = criterion(model, sample)
        full_split_loss += loss
        count += sample_size
        if count > 50000:
            break
    full_split_loss.backward()
    print(f" Lang direction : {langdir} loss : {full_split_loss.item()}")
    grads = {
        "all": grad2vec(model, modules="all") / count,
        "encoder": grad2vec(model, modules="encoder") / count,
        "encoder\emb": grad2vec(model, modules="encoder\emb") / count,
        "decoder": grad2vec(model, modules="decoder") / count,
        "decoder\emb": grad2vec(model, modules="decoder\emb") / count,
    }
    return grads


def report(similarities, langs, output_dir, label="Title"):
    labels = langs.split(",")
    M = []
    for i in labels:
        arr = []
        for j in labels:
            if i == j:
                arr.append(1.0)
            else:
                if i < j:
                    arr.append(similarities[f"{i}-{j}"])
                else:
                    arr.append(similarities[f"{j}-{i}"])
        M.append(arr)

    f, ax = plt.subplots(figsize=(32, 32))
    ax = sns.heatmap(M, linewidths=0.5, vmin=0.0, vmax=1.0)
    ax.set_yticklabels(labels, fontsize=10, color="gray")
    ax.set_xticklabels(labels, fontsize=10, color="gray")
    ax.set_title(label, fontsize=10, color="gray")
    ax.invert_yaxis()

    plt.savefig(os.path.join(output_dir, f"{label}_output.png"))


def calculate_grads(
    direction: str,
    output_dir: str,
    model_path: str,
    data_path: str,
) -> None:
    logger.info(f"Calculating gradients data for {direction}")
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [model_path], arg_overrides={"data": data_path}
    )
    model = models[0]
    criterion = task.build_criterion(cfg.criterion)
    task.load_dataset("valid")

    grads = compute_gradients(direction, task, model, criterion)
    language = direction.split("-")[-1]
    torch.save(grads, os.path.join(output_dir, f"{language}_grads.pt"))

    return "Done"


def calculate_similarity(l1: str, langs: str):
    similarities_all = {}
    similarities_enc = {}
    similarities_dec = {}
    grad_l1 = torch.load(os.path.join(args.output_dir, f"{l1}_grads.pt"))
    for l2 in tqdm.tqdm(langs):
        if l1 < l2:
            grad_l2 = torch.load(os.path.join(args.output_dir, f"{l2}_grads.pt"))
            similarities_all["{}-{}".format(l1, l2)] = cosine_similarity(
                grad_l1["all"], grad_l2["all"], dim=0
            )
            similarities_enc["{}-{}".format(l1, l2)] = cosine_similarity(
                grad_l1["encoder"], grad_l2["encoder"], dim=0
            )
            # similarities_enc_woemb["{}-{}".format(l1, l2)] = cosine_similarity(
            #     grad_enc_woemb[l1], grad_enc_woemb[l2], dim=0
            # )
            similarities_dec["{}-{}".format(l1, l2)] = cosine_similarity(
                grad_l1["decoder"], grad_l2["decoder"], dim=0
            )
            # similarities_dec_woemb["{}-{}".format(l1, l2)] = cosine_similarity(
            #     grad_dec_woemb[l1], grad_dec_woemb[l2], dim=0
            # )
    del grad_l1

    return (similarities_all, similarities_enc, similarities_dec)


async def main(args) -> None:

    langs = args.langs.split(",")

    logger.info("Start extracting gradients")
    executor = AutoExecutor(
        folder=os.path.join(args.output_dir, args.output_dir),
        cluster="slurm",
    )
    executor.update_parameters(
        slurm_partition="nllb,learnfair",
        timeout_min=1440,  # TODO: we need to tune this
        nodes=1,  # we only need one node for this
        cpus_per_task=60,
        gpus_per_node=1,
        mem_gb=512,
    )

    jobs = []
    for lang in langs:
        jobs.append(
            executor.submit(
                calculate_grads,
                f"eng-{lang}",
                args.output_dir,
                args.model_path,
                args.data_path,
            )
        )

    await asyncio.gather(*[awaitable_job(j) for j in jobs])

    logger.info("Start calculating similarities")

    # Similarities
    (
        similarities_all,
        similarities_enc,
        similarities_enc_woemb,
        similarities_dec,
        similarities_dec_woemb,
    ) = ({}, {}, {}, {}, {})

    jobs = []
    for lang in langs:
        jobs.append(
            executor.submit(
                calculate_similarity,
                lang,
                langs,
            )
        )

    results_similarities = await asyncio.gather(*[awaitable_job(j) for j in jobs])

    for i in results_similarities:
        similarities_all = {**similarities_all, **i[0]}
        similarities_enc = {**similarities_enc, **i[1]}
        similarities_dec = {**similarities_dec, **i[2]}

    torch.save(similarities_all, os.path.join(args.output_dir, f"similarities_all.pt"))
    torch.save(similarities_enc, os.path.join(args.output_dir, f"similarities_enc.pt"))
    torch.save(similarities_dec, os.path.join(args.output_dir, f"similarities_dec.pt"))

    # similarities_all = torch.load(os.path.join(args.output_dir, f"similarities_all.pt"))
    # similarities_enc = torch.load(os.path.join(args.output_dir, f"similarities_enc.pt"))
    # similarities_dec = torch.load(os.path.join(args.output_dir, f"similarities_dec.pt"))

    report(similarities_all, args.output_dir, "All")
    report(similarities_enc, args.output_dir, "Enc")
    report(similarities_dec, args.output_dir, "Dec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--langs", default=None)
    args = parser.parse_args()

    asyncio.run(main(args))
