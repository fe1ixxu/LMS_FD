#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any, Callable, Dict, List, Optional

from fb_sweep import sweep
from fb_sweep.sweep import hyperparam

GridFunc = Callable[[], List[hyperparam]]
PREDEFINED_GRIDS: Dict[str, GridFunc] = {}


def register_grid(
    name: str, *, grids: Dict[str, GridFunc] = PREDEFINED_GRIDS
) -> Callable[[GridFunc], GridFunc]:
    """A decorator to register a grid function to the given name. A grid function is a
    nullary callable which returns a list of hyperparameters.

    Currently, grids should be registered by architecture (i.e. the value passed to
    `--arch` at the command line), but this can be generalized in the future.

    Examples:
    >>> @register_grid("foo")
    >>> def foo():
    ...   return [hyperparam("--arch", "foo"), hyperparam("--foo-option", 42)]
    ...
    >>> get_predefined_grid("foo")
    [hyperparam("--arch", "foo"), hyperparam("--foo-option", 42)]

    Parameters:
        name: The name to register this grid function under.
        grids: The registry of grid functions to use.

    Raises:
        ValueError if there already exists a grid function registered under the given
            name.
    """

    def register_grid_func(fn: GridFunc) -> GridFunc:
        if name in grids:
            raise ValueError(f"There is already a grid registered under: {name}")

        grids[name] = fn
        return fn

    return register_grid_func


def get_predefined_grid(
    name: str,
    *,
    grids: Dict[str, GridFunc] = PREDEFINED_GRIDS,
) -> Optional[List[hyperparam]]:
    """Get a registered predefined grid by invoking the callable registered to
    the given name and returning the resulting list of hyperparameters.

    See also: `register_grid`

    Raises:
        KeyError if no grid is registered to the given name.
    """
    return grids[name]()


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
        "--validate-interval-updates",
        type=int,
        default=20000,
        help="Number of training updates per validation run.",
    )
    parser.add_argument(
        "--opt",
        default="adam",
        type=str,
        help="optimizer",
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
            "--encoder-langtok", encoder_langtok, save_dir_key=lambda val: f"ent{val}"
        ),
        hyperparam(
            "--sampling-method", sampling_method, save_dir_key=lambda val: f"SPL_{val}"
        ),
        hyperparam(
            "--sampling-temperature",
            sampling_temperature,
            save_dir_key=lambda val: f"tmp{val}",
        ),
        hyperparam(
            "--share-all-embeddings",
            [False],
            binary_flag=True,
            save_dir_key=lambda val: "shareemb",
        ),
        hyperparam("--adam-eps", 1e-06),
        hyperparam("--adam-betas", "(0.9, 0.98)"),
        hyperparam("--decoder-normalize-before"),
        hyperparam("--encoder-normalize-before"),
        hyperparam("--lr-scheduler", "inverse_sqrt"),
        hyperparam("--warmup-init-lr", 1e-7),
        # TODO: sweep over warmup-updates and LR
        hyperparam("--warmup-updates", [8000], save_dir_key=lambda val: f"warmup{val}"),
        hyperparam("--lr", args.lr, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--stop-min-lr", 1e-9),
        hyperparam("--clip-norm", 0.0),
        hyperparam("--dropout", args.dropout, save_dir_key=lambda val: f"drop{val}"),
        hyperparam("--weight-decay", 0.0),
        hyperparam(
            "--criterion",
            "label_smoothed_cross_entropy"
            if not args.moe
            else "moe_label_smoothed_cross_entropy",
        ),
        hyperparam("--label-smoothing", 0.1),
        hyperparam(
            "--max-tokens", args.max_tokens, save_dir_key=lambda val: f"maxtok{val}"
        ),
        hyperparam("--seed", [args.seed], save_dir_key=lambda val: f"seed{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 100),
        hyperparam(
            "--validate-interval-updates",
            args.validate_interval_updates,
        ),
        hyperparam("--keep-interval-updates", 40),
        # hyperparam("--batch-size-valid", 2),
        hyperparam(
            "--max-source-positions", 512, save_dir_key=lambda val: f"max_pos{val}"
        ),
        hyperparam("--max-target-positions", 512),
        # hyperparam("--batch-size", 16, save_dir_key=lambda val: f"bsz{val}"),
        # hyperparam("--pad-to-fixed-length"),
    ]

    # optimizer
    original_opt = args.opt
    if args.opt == "adam16bit":
        args.opt = "adam"
        grids.append(hyperparam("--fp16-adam-stats")),
    elif args.opt == "adam8bit":
        grids.append(hyperparam("--no-scale-embedding"))
        grids.append(hyperparam("--use-stable-embedding"))
        grids.append(hyperparam("--block-wise"))

    grids.append(
        hyperparam("--optimizer", args.opt, save_dir_key=lambda val: original_opt),
    )

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
                hyperparam("--moe-freq", [args.moe_freq]),
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
                save_dir_key=lambda val: "det",
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

    grids.extend(get_predefined_grid(args.arch))

    return grids


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


@register_grid("transformer")
@register_grid("transformer_wmt_en_de")
def transformer() -> List[hyperparam]:
    return [
        hyperparam(
            "--arch", ["transformer_wmt_en_de"], save_dir_key=lambda val: f"arch{val}"
        )
    ]


@register_grid("transformer_12_12_wide")
def transformer_12_12_wide() -> List[hyperparam]:
    def key(prefix: str):
        return lambda v: f"{prefix}{v}"

    return [
        hyperparam("--arch", ["transformer_wmt_en_de"], save_dir_key=key("arch")),
        hyperparam("--encoder-ffn-embed-dim", 1536 * 4, save_dir_key=key("effn")),
        hyperparam("--decoder-ffn-embed-dim", 1536 * 4, save_dir_key=key("dffn")),
        hyperparam("--encoder-embed-dim", 1536, save_dir_key=key("eem")),
        hyperparam("--decoder-embed-dim", 1536, save_dir_key=key("dem")),
        hyperparam("--encoder-layers", 12, save_dir_key=key("ne")),
        hyperparam("--decoder-layers", 12, save_dir_key=key("nd")),
    ]


@register_grid("transformer_12_12_wider")
def get_transformer_12_12_wider_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 12, save_dir_key=lambda val: f"ELS{val}"),
        hyperparam("--decoder-layers", 12, save_dir_key=lambda val: f"DLS{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            4 * 3072,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            4 * 3072,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 3072, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 3072),
        hyperparam("--encoder-attention-heads", 32, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 32),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"ATTDRP{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"RELDRP{val}"),
    ]


if __name__ == "__main__":
    sweep.main(
        get_grid, postprocess_hyperparams, add_extra_options_func=add_extra_options_func
    )
