#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fb_sweep import sweep
from fb_sweep.sweep import hyperparam

PREDIFINED_GRID_FUNCTION = {}


def register_grid(name):
    def register_grid_func(fn):
        if name not in PREDIFINED_GRID_FUNCTION:
            PREDIFINED_GRID_FUNCTION[name] = fn
        return fn

    return register_grid_func


def get_predefined_grid(name):
    if name not in PREDIFINED_GRID_FUNCTION:
        return []
    else:
        return PREDIFINED_GRID_FUNCTION[name]()


def add_extra_options_func(parser):
    parser.add_argument("--max-update", help="max update", default=40000)
    parser.add_argument(
        "--finetune-from-model",
        help="finetune from a pretrained model",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--lang-dict",
        help="a file containing a list of languages to support",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max-tokens", help="max tokens per batch", type=int, default=None
    )
    parser.add_argument("--arch", default="transformer")
    parser.add_argument("--task", default="translation_multi_simple_epoch")
    # parser.add_argument(
    #     "--langs",
    #     default=None,
    #     type=str,
    #     help="a list of languages comma sperated languages which can appear in lang-pairs; "
    #     "note that the ordering determines language token IDs",
    # )
    parser.add_argument(
        "--lang-pairs", help="lang pairs for multilingual training", type=str
    )
    parser.add_argument(
        "--sampling-method", help="sampling method", default="temperature"
    )
    parser.add_argument(
        "--sampling-temperature", help="sampling temperature", default=5
    )
    parser.add_argument(
        "--encoder-langtok", help="add src language token to encoder", default="src"
    )
    parser.add_argument("--decoder-langtok", default=True, action="store_true")
    parser.add_argument("--virtual-epoch-size", default=None)
    parser.add_argument("--virtual-data-size", default=None)
    # equivalent to training on 16x GPUs
    # parser.add_argument("--update-freq", default=16)
    # use double the default learning rate, since we're using --update-freq=16
    # per token learning should be approximately constant;
    # ideally momentent and 2nd momentent of adam should be adjusted accordingly but less important
    parser.add_argument("--lr", default=10e-4)
    parser.add_argument("--dropout", default=0.1)
    parser.add_argument(
        "--ddp-backend",
        default=None,
    )
    parser.add_argument(
        "--enable-reservsed-directions-shared-datasets",
        default=False,
        action="store_true",
    )
    parser.add_argument("--save-interval-updates", default=None)
    parser.add_argument(
        "--moe",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--moe-eval-cap",
        default=0.25,
        type=float,
        help="moe-eval-capacity-token-fraction",
    )
    parser.add_argument(
        "--moe-freq",
        default=0,
        type=int,
        help="frequency at which MoE layers exist in the Transformer",
    )
    parser.add_argument(
        "--encoder-moe-freq",
        default=0,
        type=int,
        help="frequency at which MoE layers exist in the Transformer encoder",
    )
    parser.add_argument(
        "--decoder-moe-freq",
        default=0,
        type=int,
        help="frequency at which MoE layers exist in the Transformer decoder",
    )
    parser.add_argument(
        "--log-interval",
        default=100,
        type=int,
        help="frequency at which MoE layers exist in the Transformer decoder",
    )


def get_grid(args):
    task = args.task
    sampling_method = args.sampling_method
    sampling_temperature = args.sampling_temperature
    encoder_langtok = args.encoder_langtok

    grids = [
        hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("--fp16-no-flatten-grads"),
        # TODO: verify these values
        hyperparam(
            "--max-update", [args.max_update], save_dir_key=lambda val: f"mu{val}"
        ),
        hyperparam(
            "--update-freq", [args.update_freq], save_dir_key=lambda val: f"uf{val}"
        ),  # args.update_freq),
        hyperparam("--task", task),
        hyperparam("--lang-pairs", args.lang_pairs),
        hyperparam(
            "--encoder-langtok",
            encoder_langtok,
        ),
        hyperparam(
            "--sampling-method",
            sampling_method,
        ),
        hyperparam(
            "--sampling-temperature",
            sampling_temperature,
            save_dir_key=lambda val: f"tmp{val}",
        ),
        hyperparam(
            "--share-all-embeddings",
            [True],
            binary_flag=True,
            save_dir_key=lambda val: "shareemb",
        ),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-eps", 1e-06),
        hyperparam(
            "--adam-betas",
            "(0.9, 0.98)",
        ),
        hyperparam(
            "--arch", ["transformer_wmt_en_de"], save_dir_key=lambda val: f"arch{val}"
        ),
        hyperparam("--encoder-layers", 24, save_dir_key=lambda val: f"ELS{val}"),
        hyperparam("--decoder-layers", 24, save_dir_key=lambda val: f"DLS{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--decoder-normalize-before", save_dir_key=lambda val: "dpreln"),
        hyperparam("--encoder-normalize-before", save_dir_key=lambda val: "epreln"),
        # hyperparam("--encoder-ffn-embed-dim", 8192),
        # hyperparam("--decoder-ffn-embed-dim", 8192),
        hyperparam("--lr-scheduler", "inverse_sqrt"),
        hyperparam(
            "--warmup-init-lr",
            1e-7,
        ),
        # TODO: sweep over warmup-updates and LR
        hyperparam(
            "--warmup-updates",
            [8000],
        ),
        hyperparam("--lr", args.lr, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--stop-min-lr", 1e-9),
        hyperparam(
            "--clip-norm",
            0.0,
        ),
        hyperparam(
            "--dropout",
            args.dropout,
        ),
        hyperparam(
            "--weight-decay",
            0.0,
        ),
        hyperparam(
            "--criterion",
            "label_smoothed_cross_entropy"
            if not args.moe
            else "moe_label_smoothed_cross_entropy",
        ),
        hyperparam(
            "--label-smoothing",
            0.1,
        ),
        hyperparam(
            "--max-tokens", args.max_tokens, save_dir_key=lambda val: f"maxtok{val}"
        ),
        hyperparam(
            "--seed",
            [2],
        ),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", args.log_interval),
        hyperparam("--validate-interval-updates", 20000),
        hyperparam("--keep-interval-updates", 40),
        # hyperparam("--batch-size-valid", 2),
        hyperparam("--max-source-positions", 512),
        hyperparam("--max-target-positions", 512),
        # hyperparam("--batch-size", 16, save_dir_key=lambda val: f"bsz{val}"),
        # hyperparam("--pad-to-fixed-length"),
    ]
    if args.moe:
        if args.moe_freq > 0:
            args.encoder_moe_freq = args.moe_freq
            args.decoder_moe_freq = args.moe_freq
        grids.extend(
            [
                hyperparam(
                    "--moe-gate-loss-wt", [0.01], save_dir_key=lambda val: f"moe_w{val}"
                ),
                hyperparam("--moe-gate-loss-combine-method", "sum"),
                hyperparam(
                    "--moe-second-expert-policy", ["all"], save_dir_key=lambda val: val
                ),
                hyperparam(
                    "--moe-normalize-gate-prob-before-dropping",
                    [False],
                    binary_flag=True,
                    save_dir_key=lambda val: "norm_b",
                ),
                hyperparam("--moe-gating-use-fp32"),
                hyperparam(
                    "--encoder-moe-freq",
                    args.encoder_moe_freq,
                    save_dir_key=lambda val: f"emq{val}",
                ),
                hyperparam(
                    "--decoder-moe-freq",
                    args.decoder_moe_freq,
                    save_dir_key=lambda val: f"dmq{val}",
                ),
                hyperparam("--moe-expert-count", args.num_nodes * args.num_gpus),
                hyperparam("--moe-eval-capacity-token-fraction", args.moe_eval_cap),
            ]
        )
        if args.moe_eval_cap > 0.25:
            grids.append(hyperparam("--max-tokens-valid", 1024))

    if args.ddp_backend:
        grids.append(
            hyperparam(
                "--ddp-backend", args.ddp_backend, save_dir_key=lambda val: f"{val}"
            )
        )

    if args.decoder_langtok:
        grids.append(
            hyperparam(
                "--decoder-langtok",
                [True],
                binary_flag=True,
            )
        )
    if args.virtual_data_size:
        grids.append(hyperparam("--virtual-data-size", args.virtual_data_size))
    if args.virtual_epoch_size:
        grids.append(hyperparam("--virtual-epoch-size", args.virtual_epoch_size))
    if args.lang_dict:
        grids.append(hyperparam("--lang-dict", args.lang_dict))
    if args.langs:
        grids.append(hyperparam("--langs", args.langs))
    # if args.max_tokens:
    #     grids.append(hyperparam("--max-tokens", args.max_tokens))
    if args.finetune_from_model:
        grids.append(hyperparam("--finetune-from-model", args.finetune_from_model))
    if args.enable_reservsed_directions_shared_datasets:
        grids.append(
            hyperparam(
                "--enable-reservsed-directions-shared-datasets",
                [True],
                binary_flag=True,
            )
        )
    if args.save_interval_updates:
        grids.append(
            hyperparam("--save-interval-updates", args.save_interval_updates),
        )

    return grids


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(
        get_grid, postprocess_hyperparams, add_extra_options_func=add_extra_options_func
    )
