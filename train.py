"""
THIS CODE BELONGS ALMOST ENTIRELY TO CHRIS MCCORMICK
https://colab.research.google.com/drive/1kv6G54avvUSTHi8zYXd_Lz1Gtu4VstDB?authuser=1

CODE FOUND ON:
https://github.com/KellerJordan/modded-nanogpt/discussions/156
"""

import argparse
from dataclasses import dataclass
import os
import torch
import numpy as np
import wandb
import importlib
import sys
import subprocess
import shutil
import time
import datetime as dt
from typing import Iterable, Tuple
import torch.nn as nn
import copy
import glob
import math
import threading
from collections import defaultdict
from itertools import accumulate
from pathlib import Path
import torch._dynamo as dynamo
import torch.distributed as dist
import torch.nn.functional as F
# torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
import triton
import triton.language as tl
from torch import Tensor, nn

dynamo.config.recompile_limit = 64

# region flash attention installer

def _run_pip_install(pkg_or_url):
    """Run pip install for the currently running Python executable."""
    try:
        exe = sys.executable
        cmd = [exe, "-m", "pip", "install", "--upgrade", pkg_or_url]
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        return False

def ensure_flash_attn():
    try:
        import flash_attn  # type: ignore
        print("flash_attn already installed.")
        return True
    except Exception:
        print("flash_attn not found — attempting install...")

    # Try to infer Python tag (cp310, cp311, etc.)
    py_major, py_minor = sys.version_info[:2]
    py_tag = f"cp{py_major}{py_minor}"  # e.g. cp310, cp311

    # Try to detect torch & CUDA info to pick the best matching wheel
    torch = None
    torch_ver = None
    cuda_ver = None
    try:
        import torch
        torch = torch
        torch_ver = getattr(torch, "__version__", None)
        cuda_ver = getattr(torch.version, "cuda", None)  # e.g. "11.8" or "12.4"
    except Exception:
        pass

    # Helper: map cuda version like "11.8" -> "cu118", "12.4" -> "cu124"
    def cuda_to_tag(cuda_version):
        if not cuda_version:
            return None
        # keep only digits
        parts = cuda_version.split(".")
        if len(parts) >= 2:
            return "cu" + parts[0] + parts[1]
        return "cu" + parts[0]

    # Helper: map torch.__version__ "2.6.3" -> "torch2.6"
    def torch_to_tag(torch_version):
        if not torch_version:
            return None
        # strip +cu.. if present
        base = torch_version.split("+", 1)[0]
        parts = base.split(".")
        if len(parts) >= 2:
            return "torch" + parts[0] + "." + parts[1]
        return "torch" + parts[0]

    cuda_tag = cuda_to_tag(cuda_ver)
    torch_tag = torch_to_tag(torch_ver)

    # release and base url used for prebuilt wheels (latest release checked: v0.5.4)
    release_tag = "v0.5.4"
    base_url = f"https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/{release_tag}/"

    # Wheel naming pattern appears like:
    # flash_attn-2.6.3+<cuTAG><torchTAG>-<py_tag>-<py_tag>-linux_x86_64.whl
    # We'll try a prioritized list of candidates.
    candidate_wheels = []

    # Common wheel version in release (adjustable if you want different flash_attn version)
    flash_attn_ver = "2.6.3"

    # If we detected cuda and torch, try a direct match first
    if cuda_tag and torch_tag:
        candidate_wheels.append(
            f"flash_attn-{flash_attn_ver}+{cuda_tag}{torch_tag}-{py_tag}-{py_tag}-linux_x86_64.whl"
        )

    # If we detected only cuda, try with a few likely torch tags (2.5..2.8)
    if cuda_tag and not torch_tag:
        for t in ["torch2.8", "torch2.7", "torch2.6", "torch2.5"]:
            candidate_wheels.append(
                f"flash_attn-{flash_attn_ver}+{cuda_tag}{t}-{py_tag}-{py_tag}-linux_x86_64.whl"
            )

    # If we detected only torch, try mapping to a few cuda variants (12.x then 11.x)
    if torch_tag and not cuda_tag:
        for cu in ["cu124", "cu126", "cu128", "cu130", "cu118", "cu117", "cu116"]:
            candidate_wheels.append(
                f"flash_attn-{flash_attn_ver}+{cu}{torch_tag}-{py_tag}-{py_tag}-linux_x86_64.whl"
            )

    # As a final fallback, try several common combos (covers many release assets)
    fallback_cu = ["cu124", "cu126", "cu128", "cu130", "cu118", "cu117", "cu116"]
    fallback_torch = ["torch2.8", "torch2.7", "torch2.6", "torch2.5"]
    for cu in fallback_cu:
        for t in fallback_torch:
            candidate_wheels.append(f"flash_attn-{flash_attn_ver}+{cu}{t}-{py_tag}-{py_tag}-linux_x86_64.whl")

    # Make unique while preserving order
    seen = set()
    candidate_wheels = [w for w in candidate_wheels if not (w in seen or seen.add(w))]

    # Try installing each candidate from the GitHub release URL
    for wh in candidate_wheels:
        url = base_url + wh
        print("Trying wheel:", wh)
        success = _run_pip_install(url)
        if success:
            # verify import
            try:
                import importlib
                importlib.invalidate_caches()
                flash_attn = importlib.import_module("flash_attn")
                print("Installed flash_attn from", url)
                return True
            except Exception:
                print("Installed but import failed; continuing to next candidate.")

    # Final fallback: try pip install from PyPI
    print("No matching prebuilt wheel succeeded; trying `pip install flash-attention` from PyPI")
    if _run_pip_install("flash-attention"):
        try:
            import importlib
            importlib.invalidate_caches()
            import flash_attn  # type: ignore
            print("Installed flash_attn from PyPI.")
            return True
        except Exception:
            print("flash_attn installation from PyPI failed to import.")

    print("Automatic installation failed. Please install a matching wheel for your system manually.")
    print("Repository releases with prebuilt wheels: https://github.com/mjun0812/flash-attention-prebuild-wheels/releases")
    return False

# call it early in train.py
ensure_flash_attn()
# endregion

# region utils

def summarize_parameters(model: nn.Module, display_bias: bool = True) -> int:
    """Print a table of parameter names, shapes and counts."""

    # Retrieve the list of parameters with their names.
    params: Iterable[Tuple[str, nn.Parameter]] = list(model.named_parameters())

    print0("The model has {:} different named parameters.\n".format(len(params)))

    # Print out the parameters and their shapes in table form.
    print0(
        "Parameter Name                                    Dimensions       Total Values    Trainable    Type\n"
    )

    for p_name, p in params:
        p_size = list(p.size())
        for i in range(len(p_size) - 1, -1, -1):
            if p_size[i] == 1:
                del p_size[i]
        if len(p_size) == 1:
            if not display_bias:
                continue
            p_dims = "{:>10,} x {:<10}".format(p.size()[0], "-")
        elif len(p_size) == 2:
            p_dims = "{:>10,} x {:<10,}".format(p.size()[0], p.size()[1])
        elif len(p_size) == 3:
            p_dims = "{:>10,} x {:,} x {:<10}".format(p.size()[0], p.size()[1], p.size()[2])
        elif len(p_size) == 4:
            p_dims = "{:>10,} x {:,} x {:,} x {:<10}".format(
                p.size()[0], p.size()[1], p.size()[2], p.size()[3]
            )
        else:
            print0("Unexpected: ", p.size(), p_name)
            break
        print0(
            "{:<45} {:}    {:>6}    {:}         {:}".format(
                p_name, p_dims, format_size(p.numel()), p.requires_grad, p.dtype
            )
        )

    # Tally up the total number of values.
    total_params = 0
    for _, p in params:
        total_params += p.numel()

    print0(f"\nTotal elements: {format_size(total_params)} ({total_params:,})\n")

    print0("K = 2^10 = 1,024")
    print0("M = 2^20 = 1,048,576")
    print0("B = 2^30 = 1,073,741,824")

    return total_params

def format_size(num: int) -> str:
    """Format base-2 quantities with the appropriate suffix."""

    suffixes = [" ", "K", "M", "B"] # and "T"

    base = 1024

    # Find the largest appropriate suffix.
    for suffix in suffixes:
        if abs(num) < base:
            if num % 1 != 0:
                return f"{num:.2f}{suffix}"
            else:
                return f"{num:.0f}{suffix}"
        num /= base
    if num % 1 != 0:
        return f"{num:.2f}T"
    return f"{num:.0f}T"

def fmt_elapsed(elapsed):
    """Takes a time in seconds and returns a string hh:mm:ss"""

    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(dt.timedelta(seconds=elapsed_rounded))

def print0(s, console=False):
    if master_process:
        with open(args.log_file, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# Measure the notebook's end-to-end runtime.
full_notebook_t0 = time.time()

# Small helper for timestamping.
from zoneinfo import ZoneInfo
import datetime as dt

def get_timestamp(timezone_str: str = "America/Los_Angeles") -> str:
    tz = ZoneInfo(timezone_str)
    now = dt.datetime.now(tz)
    return now.strftime("%Y-%m-%d_%H%M%S")

def get_lr(step: int):
    x = min(0.9999, step / args.num_scheduled_iterations)
    assert 0 <= x < 1
    lr = 1.0
    if x >= 1 - args.cooldown_frac:
        w = (1 - x) / args.cooldown_frac
        lr = w * 1.0 + (1 - w) * 0.1
    return lr

def get_ws(step: int):
    # set short window size to half of long window size
    # Higher ws on "extension" steps
    if step >= args.num_scheduled_iterations:
        return args.ws_final // 2, args.ws_final
    x = step / args.num_scheduled_iterations
    assert 0 <= x < 1
    ws_idx = int(len(args.ws_schedule) * x)
    return args.ws_schedule[ws_idx] // 2, args.ws_schedule[ws_idx]

def get_muon_momentum(step: int, muon_warmup_steps=300, muon_cooldown_steps=50, momentum_min=0.85, momentum_max=0.95):
    # warmup phase: linearly increase momentum from min to max
    # cooldown phase: linearly decrease momentum from max to min
    momentum_cd_start = args.num_iterations - muon_cooldown_steps
    if step < muon_warmup_steps:
        frac = step / muon_warmup_steps
        momentum = momentum_min + frac * (momentum_max - momentum_min)
    elif step > momentum_cd_start:
        frac = (step - momentum_cd_start) / muon_cooldown_steps
        momentum = momentum_max - frac * (momentum_max - momentum_min)
    else:
        momentum = momentum_max
    return momentum

# endregion utils

# region parser

parser = argparse.ArgumentParser()
parser.add_argument("language", type = str)
parser.add_argument("paradigm", type = str)
parser.add_argument("data_path", type = str)

parser.add_argument("--vocab_size", type = int, default = 50048)
parser.add_argument("--eos_id", type = int, default = 288)

parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

cli_args = parser.parse_args()

# endregion parser

# region definitions

@dataclass
class Configuration:

    # ==== Weights and Biases ====
    # Enable/Disable wandb by setting `wandb_mode` to "online"/"offline"
    wandb_mode: str        = "online"
    wandb_un: str          = "timo-liu-uc-davis-org" # <--- Update to yours
    wandb_project: str     = "tokenization_tests"
    wb_run_name: str       = f"{cli_args.language}-{cli_args.paradigm}"
    wb_run_id              = None   # Auto-set
    wandb_watch: bool      = True  # Debug run with detailed metrics.

    # ==== Model ====
    num_layers: int   = 12
    num_heads: int    = 6
    head_dim: int     = 128
    model_dim: int    = 768
    max_seq_len: int  = None  # Will be calculated based on batch size and world size

    total_params: int = None # Will be set after model is created

    # ==== Vocabulary ====
    vocab_size: int   = cli_args.vocab_size
    bos_token_id: int = cli_args.eos_id

    # ==== Data ====
    # Note: Total tokens trained is complex due to window sizes.
    num_chunks: int         = 9 # 900x10^6 tokens.
    train_batch_size: int   = 2048 * 16 * 8     # 256K
    train_max_seq_len: int  = 128 * 16          # 2K
    val_batch_size: int     = 4 * 64 * 1024 * 8 # 2M

    # Dataset settings (GPT-2 FineWeb)
    repo_id: str             = "kjj0/fineweb10B-gpt2"
    local_dir: str           = cli_args.data_path
    train_files_pattern: str = f"{cli_args.language}_{cli_args.paradigm}_CORPUS/{cli_args.language}_{cli_args.paradigm}_train_*.bin"
    val_files_pattern: str   = f"{cli_args.language}_{cli_args.paradigm}_CORPUS/{cli_args.language}_{cli_args.paradigm}_val_*.bin"
    train_files: str         = None  # Will be set from patterns
    val_files: str           = None

    val_ratio: float   = 1.0
    val_tokens: int    = 10485760

    # ==== Optimization ====
    num_scheduled_iterations: int = 2245  # number of steps to complete lr and ws schedule
    num_extension_iterations: int = 40  # number of steps to continue training at final lr and ws
    num_iterations: int = num_scheduled_iterations + num_extension_iterations

    cooldown_frac: float  = 0.50  # fraction of num_scheduled_iterations spent cooling down the learning rate
    cooldown_start: int   = None # 950  # optional / calculated.
    muon_lr               = 0.03 # Updated from 0.06
    adam_lr               = 0.008
    momentum_warmup_steps = 300 # Muon warmup 0.85 --> 0.95, cooldown 0.95 --> 0.85
    momentum_cd_steps     = 50  # number of iterations for muon momentum cooldown

    # ==== Window Scaling ====
    block_size: int          = 128
    ws_schedule: tuple       = (3,    7,   11)
    ws_schedule_steps: tuple = None # Optionally specify step transitions, e.g., (0, 450, 900)
    ws_final: int            = 13 # increase final validation ws, used for YaRN extension and short window size @classiclarryd
    ws_validate_post_yarn_ext: int = 20 # extend long windows out even further after applying YaRN

    # Define short (1) vs. long (2) context layers. (Or (0) for missing layers).
    layer_window_sizes       = [0, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 2]

    # ==== Validation and Logging ====
    # Optional: provide the path and we'll prepend the code to the log.
    #this_notebook: str    = "/content/drive/MyDrive/Colab Notebooks/Modded-NanoGPT.ipynb"
    this_notebook: str    = None

    run_id: str            = f"{get_timestamp()}"
    #run_id: str           = f"new/{uuid.uuid4()}"

    log_file: str          = ""  # Will be set after log file is created

    val_loss_every: int    = 100  # every how many steps to evaluate val loss? 0 for only at the end

    save_checkpoint: bool = True
    early_quit: int       = None  # Quit at a specific step, or `None`.
    model_path: str       = "" # Auto set when saving checkpoint.

    run_hellaswag: bool   = True  # Run HellaSwag after training

    # ==== Distributed Training ====
    # Notebook is hardcoded for single GPU
    rank: int = 0
    world_size: int = 1
    grad_accum_steps: int = 8

# endregion definitions

# region args

args = Configuration()

# accumulations steps to simulate batches
val_accum_steps = 32


os.environ['WORLD_SIZE'] = '1' # Single (node?)
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0' # This is the one gpu

# Part of the workaround to running with `dist` on a single GPU.
os.environ['MASTER_ADDR']  = '127.0.0.1'
os.environ['MASTER_PORT']  = '29500'

rank = args.rank
world_size = args.world_size
assert 8 % world_size == 0, "world_size must be a divisor of 8"
master_process = (rank == 0) # this process will do logging, checkpointing etc.
grad_accum_steps = args.grad_accum_steps // args.world_size

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.empty(
    1, device="cuda", requires_grad=True
).backward()  # prevents a bug on some systems

# Device setup
assert torch.cuda.is_available()
device = torch.device("cuda", 0)  # Hardcoded for single GPU
torch.cuda.set_device(device)
args.max_seq_len = max(args.train_batch_size, args.val_batch_size) // (args.grad_accum_steps * args.world_size)

# endregion args

# region prep logs
if master_process:
    run_id = args.run_id
    os.makedirs("logs/new", exist_ok=True)
    args.log_file = f"logs/{run_id}.txt"
    print(args.log_file)
# endregion prep logs

# region precalc

total_steps = args.num_iterations  # num_scheduled_iterations + num_extension_iterations
lr_schedule = []

for step in range(total_steps + 1):
    if step < args.num_scheduled_iterations:
        lr_schedule.append(get_lr(step))
    else:
        # Extension steps use the final lr from scheduled iterations
        lr_schedule.append(get_lr(args.num_scheduled_iterations - 1))

# Pre-calculate optimizer-specific learning rate schedules
adam_lr_schedule = [lr * args.adam_lr for lr in lr_schedule]
muon_lr_schedule = [lr * args.muon_lr for lr in lr_schedule]

ws_short_schedule = []
ws_long_schedule = []

if args.ws_schedule_steps is not None:
    # Manual step transition points
    assert len(args.ws_schedule_steps) == len(args.ws_schedule), \
        f"ws_schedule_steps length ({len(args.ws_schedule_steps)}) must match ws_schedule length ({len(args.ws_schedule)})"

    for step in range(total_steps + 1):
        if step >= args.num_scheduled_iterations:
            # Extension steps use ws_final
            ws_short_schedule.append(args.ws_final // 2)
            ws_long_schedule.append(args.ws_final)
        else:
            # Find the appropriate ws_idx based on step transition points
            ws_idx = 0
            for i in range(len(args.ws_schedule_steps) - 1, -1, -1):
                if step >= args.ws_schedule_steps[i]:
                    ws_idx = i
                    break
            ws_idx = min(ws_idx, len(args.ws_schedule) - 1)

            ws_long = args.ws_schedule[ws_idx]
            ws_short = ws_long // 2
            ws_short_schedule.append(ws_short)
            ws_long_schedule.append(ws_long)
else:
    # Default: use get_ws() for linear division
    for step in range(total_steps + 1):
        ws_short, ws_long = get_ws(step)
        ws_short_schedule.append(ws_short)
        ws_long_schedule.append(ws_long)

momentum_schedule = []

for step in range(total_steps + 1):
    momentum_schedule.append(get_muon_momentum(step, args.momentum_warmup_steps, args.momentum_cd_steps))

# endregion precalc

# region wandbsetup

val_loss_history = [float('nan')] * (total_steps + 1)  # NaN for steps without validation
step_time_history = [0.0] * (total_steps + 1)  # Time taken for each training step (ms)
cumulative_train_time_history = [0.0] * (total_steps + 1)  # Cumulative training time up to each step (ms)
if master_process:
    run_id = args.run_id
    os.makedirs("logs/new", exist_ok=True)
    args.log_file = f"logs/{run_id}.txt"
    print(args.log_file)

def wb_log0(data, step):
    if master_process:
        wandb.log(data, step=step)

wandb.login()

# endregion wandbsetup

# region A100 magic
cc_major, cc_minor = torch.cuda.get_device_capability()

# FA3 requires an H100, because it's not built for ARM (GH200)
use_fa3 = False

if cc_major >= 9 and use_fa3:
    from kernels import get_kernel

    print0("Using FlashAttention3")
    flash_attn_interface = get_kernel('varunneal/flash-attention-3').flash_attn_interface

    print0("Running with 8-bit precision for LM-head")
    os.environ["DISABLE_FP8"] = "False"

# Lambda GH200 defaults:
#      Python: 3.10.12 (main, May 27 2025, 17:12:29) [GCC 11.4.0]
#     PyTorch: 2.7.0
#        CUDA: 12.8
#   FlashAttn: 2.7.4.post1
elif cc_major >= 9:

    print0("Using FlashAttention2")
    from flash_attn import flash_attn_interface

    print0("Running with 8-bit precision for LM-head")
    os.environ["DISABLE_FP8"] = "False"

# Colab A100:
elif cc_major < 9:
    print0("Using FlashAttention2")
    from flash_attn import flash_attn_interface

    print0("Running with BF16 for LM-head")
    os.environ["DISABLE_FP8"] = "True"

# endregion A100 magic

# region dataset settings
args.train_files = os.path.join(cli_args.data_path, args.train_files_pattern)
args.val_files = os.path.join(cli_args.data_path, args.val_files_pattern)
# endregion dataset settings

# region customops
@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]

@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def XXT_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def XXT(A: torch.Tensor, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    XXT_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
    )
    return out

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ba_plus_cAA_kernel(
    A_ptr, C_ptr,
    M,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    # This is mostly duplicated from XXT_kernel, but also loads and adds a block of A
    # Performance is slightly slower than XXT_kernel, so we use two separate kernels
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A to add (corresponds to the current block of C)
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    # Apply alpha and beta
    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = alpha * A @ A.T + beta * A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K, "Input matrix must be square"
    assert out.size(-2) == M
    assert out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ba_plus_cAA_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
    )
    return out


# Computed for num_iters=5, safety_factor=2e-2, cushion=2
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

@torch.compile(dynamic=False, fullgraph=True) # Must use dynamic=False or else it's much slower
def polar_express(G: torch.Tensor):
    """
    Polar Express Sign Method: https://arxiv.org/pdf/2505.16932
    by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.
    """
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)

    # Allocate buffers
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Perform the iterations
    for a, b, c in polar_express_coeffs:
        XXT(X, out=A)  # A = X @ X.mT
        ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A
        aX_plus_BX(X, B, X, beta=a, out=C)  # C = a * X + B @ X
        X, C = C, X  # Swap references to avoid unnecessary copies

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
# endregion customops

# region normmuon

# -----------------------------------------------------------------------------
# NorMuon optimizer

class NorMuon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).

    Differences from standard Muon:
    - Newton-Shulz is replaced with Polar Express for the orthogonalization step
    - NorMuon adds a low-rank variance estimator similar to Adafactor.
    - small 1D parameters handled via magnitude normalization of the grad (faster execution than Adam)
    - Custom distributed sizing:
    The model stores all attn and mlp weights in the same shape, and then updates the view as
    needed on the forward pass. This enables attn and mlp weights to be contained within the same
    dist.reduce_scatter_tensor() call. The model architecture has been customized to enable
    (n_attn_layers+n_mlp_layers*2)%8==0 for batching across 8 GPUs with zero padding on mlp and attn.
    The scheduling is:
        1. reduce scatter smear_gate (1 param 7 padding params)
        2. reduce scatter attn_gate (10 params 6 padding params)
        3. reduce scatter attn/mlp round 1 (10 attn params 6 mlp params)
        4. reduce scatter attn/mlp round 2 (16 mlp params)
        5. wait on step 1, then compute update of 1 and schedule all gather
        6. wait on step 2, then compute update of 2 and schedule all gather
        7. wait on step 3, then compute update of 3 and schedule all gather
            GPUs receive [2 ATTN, 2 ATTN, 2 ATTN, 2 ATTN, 2 ATTN, 2 MLP, 2 MLP, 2 MLP]
            GPUs that receive params of type attn reshape before computing update
        8. wait on 4, then compute update of 4 and schedule all gather
        9. wait for each all gather to complete and update params
    Empirically, leading with small params provides an additional 0.2s improvement.
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, beta2=0.95, custom_sizing=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        # custom sizing requires 8 GPUs
        if custom_sizing and dist.get_world_size()==8:
            param_groups = self.generate_custom_param_groups(params)
        else:
            param_groups = self.generate_standard_param_groups(params)
        super().__init__(param_groups, defaults)

    def reset(self):
        # expose a reset for clearing buffers
        for group in self.param_groups:
            group["momentum_buffer"].zero_()
            group["second_momentum_buffer"].zero_()

    def generate_standard_param_groups(self, params):
        """
        Use this method if running on less than 8 GPU or experimenting with additional attn or mlp modules.
        Creates one param group per module.
        """
        groups = defaultdict(list)
        for param in params:
            groups[param.label].append(param)

        param_groups = []
        for module_name, group_params in groups.items():
            chunk_size = (len(group_params) + self.world_size - 1) // self.world_size
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))

        return param_groups

    def generate_custom_param_groups(self, params):
        """
        Implementation requires that a single GPU does not receive both attn
        and mlp params when a param group is split across GPUs.
        """
        module_group_order = ['smear_gate', 'attn_gate', 'attn', 'mlp']
        params_list = list(params)
        params_list.sort(key=lambda x: module_group_order.index(x.label))

        idx = 0
        group_sizes = [1, 10, 16, 16]
        assert len(params_list) == sum(group_sizes)
        param_groups = []
        for size in group_sizes:
            chunk_size = (size + self.world_size - 1) // self.world_size
            group_params = params_list[idx: idx + size]
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))
            idx += size

        return param_groups

    @torch.no_grad()
    def step(self):
        # Efficient systems-wise implementation of step developed by @YouJiacheng,
        # @KonstantinWilleke, @alexrgilbert, @adricarda, @tuttyfrutyee, @vdlad,
        # @ryanyang0, @vagrawal, and @varunneal.
        rank = dist.get_rank()
        group_infos = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            if not params:
                continue

            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * self.world_size

            stacked_grads = torch.empty(
                (padded_num_params, *params[0].shape),
                dtype=params[0].dtype,
                device=params[0].device
            )
            for i, p in enumerate(params):
                stacked_grads[i].copy_(p.grad, non_blocking=True)
            if len(params) < padded_num_params:
                stacked_grads[len(params):].zero_()

            grad_chunk = torch.empty_like(stacked_grads[:chunk_size])

            reduce_future = dist.reduce_scatter_tensor(
                grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
            ).get_future()

            group_infos.append(dict(grad_chunk=grad_chunk, reduce_future=reduce_future))

        all_gather_infos = []
        # Second pass: wait for gradients, compute updates for the local shard of parameters,
        # and launch all async all_gather operations.
        for group, info in zip(self.param_groups, group_infos):
            info["reduce_future"].wait()

            params = group["params"]
            grad_chunk = info["grad_chunk"]
            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * self.world_size

            start_idx = rank * chunk_size
            module_idx = start_idx if start_idx < len(params) else 0

            num_params = min(chunk_size, max(0, len(params) - start_idx))  # num params for this rank

            if "momentum_buffer" not in group:
                group["momentum_buffer"]  = torch.zeros_like(grad_chunk[:num_params])
            momentum_buffer = group["momentum_buffer"]
            # Apply momentum update to the persistent momentum buffer in-place
            momentum_buffer.lerp_(grad_chunk[:num_params], 1 - group["momentum"])
            updated_grads = grad_chunk[:num_params].lerp_(momentum_buffer, group["momentum"])

            grad_shape = updated_grads.shape
            if params[module_idx].label == 'attn':
                # Reshape attn params from [hdim, dim*4] to [4,hdim,dim]
                for p in params[module_idx:module_idx + num_params]:
                    assert p.label == 'attn'
                updated_grads = updated_grads.view(4 * grad_shape[0], grad_shape[1], grad_shape[2] // 4)
            ref_param = params[module_idx]
            param_shape = ref_param.shape

            if "second_momentum_buffer" not in group:
                group["second_momentum_buffer"] = (torch.zeros_like(updated_grads[..., :, :1])
                    if param_shape[-2] >= param_shape[-1] else torch.zeros_like(updated_grads[..., :1, :])
                )
            second_momentum_buffer = group["second_momentum_buffer"]

            if "param_lr" not in group:
                group["param_lr"] = (
                    max(1., param_shape[-2] / param_shape[-1]) ** 0.5
                    * ref_param.new_tensor(
                        [getattr(param, "lr_mul", 1.0) for param in params[module_idx:module_idx + num_params]]
                    ).view(-1, 1, 1)
                )

                group["param_wd"] = ref_param.new_tensor(
                    [getattr(param, "wd_mul", 1.0) for param in params[module_idx:module_idx + num_params]]
                ).view(-1, 1, 1)

            # Determine LR and WR
            eff_lr = group["lr"] * group["param_lr"]
            eff_wd = group["weight_decay"] * group["param_wd"]

            # Compute zeropower for the entire chunk in a single, batched call.
            if num_params == 0:
                v_chunk = updated_grads
            else:
                v_chunk = polar_express(updated_grads)

            # NorMuon: second_momentum_buffer tracks squared magnitude of gradients along one dim (https://arxiv.org/pdf/2510.05491)
            v_norm = v_chunk.norm(dim=(-2, -1), keepdim=True)
            v_mean = v_chunk.square().mean(dim=-1 if param_shape[-2] >= param_shape[-1] else -2, keepdim=True)
            second_momentum_buffer.lerp_(v_mean.to(dtype=ref_param.dtype), 1 - group["beta2"])
            step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
            v_chunk.mul_(step_size)
            v_norm_new = v_chunk.norm(dim=(-2, -1), keepdim=True)
            v_chunk.mul_(v_norm / v_norm_new.clamp_min_(1e-10))

            v_chunk = v_chunk.view(grad_shape)

            updated_params = torch.empty_like(grad_chunk)
            param_chunk = torch.stack(params[module_idx:module_idx + num_params]) if num_params > 0 else torch.zeros_like(v_chunk)
            # Apply weight decay directly to the buffer.
            param_chunk.mul_(1 - eff_wd)

            param_chunk.add_(-eff_lr * v_chunk)

            updated_params[:num_params].copy_(param_chunk)
            if num_params < chunk_size:
                updated_params[num_params:].zero_()

            stacked_params = torch.empty(
                (padded_num_params, *param_shape),
                dtype=updated_params.dtype,
                device=updated_params.device,
            )

            gather_future = dist.all_gather_into_tensor(
                stacked_params, updated_params, async_op=True
            ).get_future()

            all_gather_infos.append(
                {
                    "gather_future": gather_future,
                    "stacked_params": stacked_params,
                    "orig_params": params,
                }
            )

        # Final pass: wait for all_gather to complete and copy results back into original parameter tensors.
        for info in all_gather_infos:
            info["gather_future"].wait()
            stacked_params = info["stacked_params"]
            orig_params = info["orig_params"]

            unstacked_params = torch.unbind(stacked_params)
            for i, p in enumerate(orig_params):
                p.copy_(unstacked_params[i], non_blocking=True)
# endregion normmuon

# region distadam
class DistAdam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)
        # init state
        for p in params:
            chunk_size = p.size(0) // self.world_size
            exp_avg = torch.zeros_like(p[:chunk_size], dtype=torch.bfloat16, device=p[0].device)
            exp_avg_sq = torch.zeros_like(exp_avg)
            self.state[p] = dict(step=0, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)
        # DistributedAdam implementation by @vagrawal

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        reduce_scatter_futures: list[torch.Future] = []
        all_gather_futures: list[torch.Future] = []
        grad_slices = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for param in params:
                grad = param.grad
                rank_size = grad.shape[0] // self.world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for param in params:
                reduce_scatter_futures[idx].wait()
                rank_size = param.shape[0] // self.world_size
                p_slice = param[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(param, "lr_mul", 1.0)
                state = self.state[param]
                g_slice = grad_slices[idx]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(param, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (bias2 ** 0.5 / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1
                all_gather_futures.append(dist.all_gather_into_tensor(param, p_slice, async_op=True).get_future())
        torch.futures.collect_all(all_gather_futures).wait()
# endregion distadam

# region model
def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()  # @Grad62304977 and others

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

# yarn implementation @classiclarryd

class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device=device)
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//4)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        theta = torch.outer(t, angular_freq)
        self.cos = nn.Buffer(
            theta.cos().to(torch.bfloat16), persistent=False
        )
        self.sin = nn.Buffer(
            theta.sin().to(torch.bfloat16), persistent=False
        )
        self.angular_freq = angular_freq
        # start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = args.block_size * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.cos.copy_(theta.cos())
        self.sin.copy_(theta.sin())
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

def rotary(x_BTHD: Tensor, cos: Tensor, sin: Tensor):
    assert cos.size(0) >= x_BTHD.size(-3)
    cos, sin = (
        cos[None, : x_BTHD.size(-3), None, :],
        sin[None, : x_BTHD.size(-3), None, :],
    )
    x1, x2 = x_BTHD.chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), 3)

@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    cos: torch.Tensor
    sin: torch.Tensor
    attn_scale: float

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim

        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        std = 0.5 * (self.dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        # make matrices the same shape as MLP to enable batched call in optimizer
        self.qkvo_w = nn.Parameter(torch.empty(self.hdim, self.dim*4))
        # label module to enable custom optimizer sizing
        self.qkvo_w.label='attn'

        with torch.no_grad():
            self.qkvo_w.view(4,self.hdim, self.dim)[:3].uniform_(-bound, bound) # init QKV weights
            self.qkvo_w.view(4,self.hdim, self.dim)[3].zero_() # init output weights to zero

        # sparse gated attention to enable context based no-op by @classiclarryd
        self.attn_gate = CastedLinear(12, num_heads)
        # label module to enable custom optimizer sizing
        self.attn_gate.weight.label = 'attn_gate'

    def forward(self, x: Tensor, attn_args: AttnArgs):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        # unpack attention args
        cos, sin = attn_args.cos, attn_args.sin
        ve, sa_lambdas = attn_args.ve, attn_args.sa_lambdas
        seqlens, attn_scale, bm_size = attn_args.seqlens, attn_args.attn_scale, attn_args.bm_size

        q, k, v = F.linear(x, self.qkvo_w.view(4, self.hdim, self.dim)[:3].flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        if ve is not None:
            v = sa_lambdas[0] * v + sa_lambdas[1] * ve.view_as(v) # @ KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = sa_lambdas[0] * v

        max_len = args.train_max_seq_len if self.training else (args.val_batch_size // (grad_accum_steps * world_size))

        # use flash_attn over flex_attn @varunneal. flash_attn_varlen suggested by @YouJiacheng
        y = flash_attn_interface.flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,
                                                        max_seqlen_q=max_len, max_seqlen_k=max_len,
                                                        causal=True, softmax_scale=attn_scale, window_size=(bm_size, 0))
        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, self.qkvo_w.view(4, self.hdim, self.dim)[3].type_as(y))
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        # make matrices the same shape to enable batched call in optimizer
        self.c_fc = nn.Parameter(torch.empty(dim, hdim))
        self.c_proj = nn.Parameter(torch.empty(dim, hdim))
        # label modules to enable custom optimizer sizing
        self.c_fc.label = 'mlp'
        self.c_proj.label = 'mlp'
        # corrective factor to account for transpose
        self.c_fc.lr_mul = 2.

        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = F.linear(x, self.c_fc.T.type_as(x))
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = F.linear(x, self.c_proj.type_as(x))
        return x

class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, head_dim, num_heads) if layer_idx not in [0, 7] else None
        # skip MLP blocks for first MLP layer by @EmelyanenkoK
        self.mlp = MLP(dim) if layer_idx != 0 else None

    def forward(self, x: Tensor, x0: Tensor, lambdas: Tensor, attn_args: AttnArgs):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), attn_args)
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.smear_gate = CastedLinear(12, 1)
        # label modules to enable custom optimizer sizing
        self.smear_gate.weight.label = 'smear_gate'
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, head_dim, num_heads, i) for i in range(num_layers)])
        self.yarn = Yarn(head_dim, max_seq_len)
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        use_fp8 = not os.environ.get("DISABLE_FP8", False)
        self.lm_head = CastedLinear(model_dim, vocab_size, use_fp8=use_fp8, x_s=(model_dim**0.5)/448, w_s=2**-9, grad_s=1/448)
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        pad = (-num_layers * 5 - 2) % dist.get_world_size()
        self.scalars = nn.Parameter(
            torch.cat(
                [
                    -1.5
                    * torch.ones(num_layers),  # skip_weights -> σ(-1.5) ≈ 0.18
                    *[
                        torch.tensor([1.0, 0.0]) for _ in range(num_layers)
                    ],  # block lambdas
                    *[
                        torch.tensor([0.5, 0.5]) for _ in range(num_layers)
                    ],  # SA lambdas
                    torch.zeros(1), # smear_lambda
                    0.5*torch.ones(1), # backout_lambda
                    torch.ones(pad),
                ]
            )
        )
        # set learning rates
        for param in self.embed.parameters():
            param.lr_mul = 75.
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.
        self.lm_head.weight.lr_mul = 1.0
        self.scalars.lr_mul = 5.0

    #def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor, ws_short: int, ws_long: int):
    def forward(self, input_seq: Tensor, seqlens: Tensor, ws_short: int, ws_long: int, target_seq: Tensor = None, inference: bool = False):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        # dropping first layer updates this to .12 ... 012
        #ve = [None, ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        #ve = [None, ve[0], ve[1], ve[2], None, None, None, None, None, ve[0], ve[1], ve[2]]
        #        0,     1,     2,     3,    4,    5,    6,    7,    8,     9,    10,    11
        ve = [None, ve[1], ve[2], None, None, None, None, None, None, ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        #short_bm = ws_short * args.block_size
        #long_bm = ws_long * args.block_size
        #bm_sizes = [None, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, None, short_bm, short_bm, short_bm, long_bm]

        # Convert window size definitions to absolute sizes.
        bm_sizes = []
        for layer_ws in args.layer_window_sizes:
            if layer_ws == 0: # 0 = no mask
                bm_sizes.append(None)
            elif layer_ws == 1: # 1 = short window
                bm_sizes.append(ws_short * args.block_size)
            elif layer_ws == 2: # 2 = long window
                bm_sizes.append(ws_long * args.block_size)

        assert len(bm_sizes) == len(self.blocks)

        x = self.embed(input_seq)

        skip_weights = self.scalars[:(len(self.blocks) // 2)]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)
        smear_lambda = self.scalars[5 * len(self.blocks)]
        backout_lambda = self.scalars[5 * len(self.blocks)+1]

        # smear token embed forward 1 position @classiclarryd
        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])

        # U-net design by @brendanh0gan
        skip_connections = []
        n = len(self.blocks) // 2

        x_backout = None
        backout_layer = 8
        # skip layer zero
        for i in range(1,len(self.blocks)):
            attn_args = AttnArgs(
                ve=ve[i],
                sa_lambdas=sa_lambdas[i],
                seqlens=seqlens,
                bm_size=bm_sizes[i],
                cos=self.yarn.cos,
                sin=self.yarn.sin,
                attn_scale=self.yarn.attn_scale
            )
            # since layer 0 is skipped, layer 11 does not have skip_connection
            if i >= n and i<11:
                gate = torch.sigmoid(skip_weights[i - n])  # in (0, 1)
                x = x + gate * skip_connections.pop()
            x = self.blocks[i](x, x0, lambdas[i], attn_args)
            if i < n:
                skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        # back out contributions from first 8 layers that are only required for downstream context and not direct prediction
        x -= backout_lambda * x_backout
        x = norm(x)
        logits = self.lm_head(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits / 7.5)

        if inference:
            return logits

        logits_for_loss = logits.float() if not self.training else logits
        loss = F.cross_entropy(
            logits_for_loss.view(-1, logits_for_loss.size(-1)),
            target_seq,
            reduction="sum" if self.training else "mean",
        )

        return loss

# endregion model

# region ddl

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

class BOSFinder:
    # Helper for getting sequences that start at the beginning of documents by @varunneal based on work by @classiclarryd
    def __init__(self, tokens: Tensor, world_size: int = 1, quickload: bool = False):
        # Precompute BOS positions once per shard
        self.tokens=tokens
        self.size = tokens.numel()
        self.quickload = quickload
        if quickload:
            # only scan first 4 million tokens, then kickoff async thread to scan rest
            self.bos_idx = (tokens[:4_000_000] == args.bos_token_id).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
            self.thread = None
            self.ready = threading.Event()
            self.start()
        else:
            self.bos_idx = (tokens == args.bos_token_id).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self.i = 0
        self.world_size = world_size
        self.batch_iter = 0

    def _load(self):
        self.bos_idx_async = (self.tokens == args.bos_token_id).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self.ready.set()

    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        self.bos_idx = self.bos_idx_async

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        # if quickload was used, repoint to the full dataset after 5 batches
        if self.quickload and self.batch_iter == 5:
            self.get()

        if max_seq_len <= 0 or num_tokens_local <= 0:
            raise ValueError(f"Invalid arguments: {num_tokens_local=}, {max_seq_len=}")

        n = len(self.bos_idx)
        if n == 0:
            raise StopIteration("No BOS tokens found in shard; advancing to next shard.")

        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]

        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    last = self.bos_idx[n - 1] if n > 0 else -1
                    raise StopIteration(
                        f"Insufficient BOS after index {int(last)}; hit tail of shard. "
                        f"(requested {num_tokens_local=}, {max_seq_len=})"
                    )

                cur = int(self.bos_idx[idx])
                starts[r].append(cur)

                next_boundary = int(self.bos_idx[idx + 1]) if (idx + 1) < n else int(self.size)
                end = min(
                    next_boundary,                         # stop at next doc start
                    cur + int(max_seq_len),                # or max sequence length
                    cur + int(num_tokens_local - cur_len + 1)  # or remaining budget (+1 for targets shift)
                )
                ends[r].append(end)

                cur_len += (end - cur)
                idx += 1

            # Expect exactly +1 because inputs/targets are offset by one token
            assert cur_len == num_tokens_local + 1, (
                f"Batch assembly mismatch: got {cur_len}, expected {num_tokens_local + 1}"
            )

        self.i = idx
        self.batch_iter += 1
        return starts, ends

def next_batch(self, num_tokens_local: int, max_seq_len: int):
    # if quickload was used, repoint to the full dataset after 5 batches
    if self.quickload and self.batch_iter==5:
        self.get()
    n = len(self.bos_idx)
    starts = [[] for _ in range(self.world_size)]
    ends = [[] for _ in range(self.world_size)]

    idx = self.i
    for r in range(self.world_size):
        cur_len = 0
        while cur_len <= num_tokens_local:
            if idx >= n:
                raise StopIteration(f"Insufficient BOS ahead of position {cur}; hit tail of shard.")
            cur = self.bos_idx[idx]
            starts[r].append(cur)
            end = min(self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                        cur + max_seq_len,
                        cur + num_tokens_local - cur_len + 1)
            ends[r].append(end)
            cur_len += end - cur
            idx += 1

        assert cur_len == num_tokens_local + 1
    self.i = idx
    self.batch_iter+=1
    return starts, ends

class DataPreloader:
    # Helper for asynchronously loading next shard and indexing bos tokens
    def __init__(self, file_iter, world_size: int = 1):
        """
        file_iter: an iterator that yields filenames (strings)
        world_size: number of processes / ranks for BOSFinder
        """
        self.file_iter = file_iter
        self.world_size = world_size
        self.thread = None
        self.data = None
        self.ready = threading.Event()

    def _load(self):
        """
        Thread target: peek header (validate) then load tokens and build BOSFinder.
        """
        try:
            fname = next(self.file_iter)
        except StopIteration:
            # no more files to load
            self.data = None
            self.ready.set()
            return

        # preserve the header-peek/validation step used in the first preloader
        # _peek_data_shard will perform the header checks (magic/version) and
        # return the claimed ntok (and will exit/assert if header invalid).
        ntok = _peek_data_shard(fname)

        # Optionally you could use ntok for additional checks here.
        # Now load actual tokens (this will skip the header internally).
        tokens = _load_data_shard(fname)

        # Sanity: loaded tokens length should match ntok
        if len(tokens) != int(ntok):
            # raise or handle as appropriate for your application
            raise RuntimeError(f"token count mismatch for {fname}: header {ntok} vs loaded {len(tokens)}")

        # Build BOSFinder and store the result
        self.data = (tokens, BOSFinder(tokens, self.world_size))
        self.ready.set()

    def start(self):
        """
        Kick off asynchronous load of the *next* file from file_iter.
        """
        # reset the ready event and start the loader thread
        self.ready.clear()
        self.thread = threading.Thread(target=self._load, daemon=True)
        self.thread.start()

    def get(self):
        """
        Wait for the async load to finish and return the (tokens, BOSFinder) tuple,
        or None if there were no files to load.
        """
        if self.thread:
            self.ready.wait()
            self.thread.join()
        return self.data

def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, grad_accum_steps: int = 1, align_to_bos: bool = True):
    # align_to_bos: each sequence begins with Beginning of Sequence token, sequences truncated to max_seq_len
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert num_tokens % (world_size * grad_accum_steps) == 0, "Batch size must be divisible by world size"
    num_tokens = num_tokens // grad_accum_steps

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = iter(files)  # Use itertools.cycle(files) for multi-epoch training
    tokens = _load_data_shard(next(file_iter))
    if align_to_bos:
        finder = BOSFinder(tokens, world_size=world_size, quickload=True)
        preloader = DataPreloader(file_iter, world_size)
        preloader.start()
    else:
        pos = 0  # for unaligned case

    while True:
        num_tokens_local = num_tokens // world_size
        max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)  # median doc length is ~400

        if align_to_bos:
            try:
                seq_starts, seq_ends = finder.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])
            except StopIteration:
                # This shard is exhausted, load the next one in the next loop iteration.
                tokens, finder = preloader.get()
                preloader.start()
                continue

            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1  # last document was too long to account for _targets offset
            cum_lengths = (end_idxs - start_idxs).cumsum(0)

        else:
            if pos + num_tokens + 1 >= len(tokens):  # should not occur for val data
                tokens, pos = _load_data_shard(next(file_iter)), 0

            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local, )
            _targets = buf[1:].view(num_tokens_local, )

            cum_lengths = torch.nonzero(_inputs == args.bos_token_id)[:, 0]
            pos += num_tokens

        num_docs = min(len(cum_lengths), max_num_docs - 1)
        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths[:num_docs]

        new_params = yield (
            _inputs.to(device="cuda", dtype=torch.int32, non_blocking=True),
            _targets.to(device="cuda", dtype=torch.int64, non_blocking=True),
            _cum_lengths.to(device="cuda", dtype=torch.int32, non_blocking=True)
        )

        if new_params is not None:
            # makes it possible for generator to receive new (num_tokens, max_seq_len, grad_accum_steps) via .send()
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * grad_accum_steps) == 0, "Num tokens must be divisible by world size"
            num_tokens = new_num_tokens
            max_seq_len = new_max_seq_len
            grad_accum_steps = new_grad_accum_steps
# endregion ddl

# region training
# -----------------------------------------------------------------------------
# int main

if not dist.is_initialized():
    # set up DDP (distributed data parallel). torchrun sets this env variable
    dist.init_process_group(backend="nccl", device_id=device)

dist.barrier()

model: nn.Module = GPT(
    vocab_size  = args.vocab_size,
    num_layers  = args.num_layers,
    num_heads   = args.num_heads,
    head_dim    = args.head_dim,
    model_dim   = args.model_dim,
    max_seq_len = args.max_seq_len
).cuda()

# Continue logging system information
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0(f"Running Triton version {triton.__version__}")

def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout

print0(nvidia_smi())
print0("="*100)

from tabulate import tabulate

print0("\n======== Configuration ========\n")

# Retrieve the args as rows and print with tabulate.
args_dict = vars(args)
rows = [(k, v) for k, v in args_dict.items()]
print0(tabulate(rows, headers=["Arg", "Value"], tablefmt="github"))

# Print the architecture
print0("\n======== Model Architecture ========\n")
print0(model)

# Print the list of parameters and their sizes
print0("\n======== Parameter Summary ========\n")
args.total_params = summarize_parameters(model)

# Add the parameter count as a prefix to the run ID.
#args.run_name = args.run_name.replace("<total_params - >", f"{format_size(args.total_params)} - ")

if master_process:
    print0("\n======== Weights & Biases ========\n")

    # ==== wandb ====
    wandb.init(
        project=args.wandb_project,
        name=args.wb_run_name,
        config=args
    )

    # Save for retrieval down in the results section.
    args.wb_run_id = wandb.run.id

    wandb.define_metric("final/*", step_metric=None, summary="last")  # one-off scalars

    if args.wandb_watch:
        # log="all" → logs both gradients and parameters
        #     "gradients" or "parameters" → limits scope
        # log_freq=100 → upload histograms every 100 steps (can be expensive)
        # log_graph=True → explicitly log model graph once (default: True for first watch call)
        wandb.watch(model, log="all", log_freq=100)


def save_checkpoint(step, reason=""):
    """
    Save a checkpoint at the current step.

    Args:
        step: Current training step
        reason: Optional description (e.g., "early_quit", "final")
    """
    if master_process and args.save_checkpoint:
        print0(f"Saving checkpoint at step {step}" + (f" ({reason})" if reason else ""))

        log = dict(
            step=step,
            code="",
            model=model.state_dict(),
            optimizers=[opt.state_dict() for opt in optimizers],
            # Schedules
            adam_lr_schedule=adam_lr_schedule,
            muon_lr_schedule=muon_lr_schedule,
            momentum_schedule=momentum_schedule,
            ws_short_schedule=ws_short_schedule,
            ws_long_schedule=ws_long_schedule,
            # Tracking arrays
            val_loss_history=val_loss_history,
            step_time_history=step_time_history,
            cumulative_train_time_history=cumulative_train_time_history,
            # Config info
            config=vars(args),
        )
        os.makedirs(f"logs/{run_id}", exist_ok=True)
        torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")

        # Set model path for potential HellaSwag evaluation
        args.model_path = f"logs/{run_id}/state_step{step:06d}.pt"

        print0(f"Checkpoint saved: {args.model_path}")

for m in model.modules():
    if isinstance(m, (nn.Embedding, nn.Linear)):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if
                        p.ndim >= 2 and "embed" not in n and "gate" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]
gate_params = [p for n, p in model.named_parameters() if "gate" in n]

# init the optimizer(s)
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer1 = DistAdam(
    scalar_params + head_params + embed_params,
    lr=args.adam_lr,
    betas=(0.65, 0.95),
    eps=1e-8,
    weight_decay=0.0,
)
optimizer2 = NorMuon(
    hidden_matrix_params + gate_params,
    lr=args.muon_lr,
    momentum=0.95,
    beta2=0.95,
    weight_decay=0.0,
)
optimizers = [optimizer1, optimizer2]

# Set initial_lr for the new step_optimizers function
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: stable then decay

def step_optimizers(step: int, optimizers, model):
    # update lr
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)

    # set muon momentum based on step
    momentum = get_muon_momentum(step)
    for group in optimizers[1].param_groups:
        group["momentum"] = momentum

    # on even steps, only step Muon params
    # on odd steps, step all params
    if step%2==0:
        optimizers[1].step()
        optimizers[1].zero_grad(set_to_none=True)
    else:
        for optimizer in optimizers:
            optimizer.step()
        model.zero_grad(set_to_none=True)

#model: nn.Module = torch.compile(model, dynamic=False, fullgraph=True)

########################################
#            Warmup kernels            #
########################################
print0("\n======== Kernel Warmup ========\n")
warmup_t0 = time.time()

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 30
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
train_loader = distributed_data_generator(
    args.train_files,
    args.train_batch_size,
    args.train_max_seq_len,
    grad_accum_steps=grad_accum_steps
)
ws_long = args.ws_schedule[0]
for step in range(warmup_steps):
    inputs, targets, cum_seqlens = next(train_loader)
    new_ws_long = args.ws_schedule[step % len(args.ws_schedule)]  # each window size is a new graph, need to warm up each with YaRN params
    if new_ws_long > ws_long:
        model.yarn.apply(ws_long, new_ws_long)
        ws_long = new_ws_long
    elif new_ws_long<ws_long:
        model.yarn.reset()
        ws_long = new_ws_long
    #model(inputs, targets, cum_seqlens, ws_long//2, ws_long).backward()
    model(inputs, cum_seqlens, ws_long//2, ws_long, targets, inference=False).backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.yarn.reset() # rotary buffer is not stored in state_dict
model.load_state_dict(initial_state["model"])
optimizer2.reset()  # Reset NorMuon momentum buffers (not stored in state dict)
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del train_loader, initial_state

warmup_time = time.time() - warmup_t0
print0(f"Warmup time: {fmt_elapsed(warmup_time)}")

# Close the wandb run if the code crashes / is interrupted.
import atexit
if master_process:
    atexit.register(wandb.finish)

print0("\n======== Training ========\n")

# The benchmark stops the clock for the validation segments. To measure how much
# these contribute, we'll measure the end-to-end training time.
train_plus_val_t0 = time.time()

########################################
#        Training and validation       #
########################################

train_loader = distributed_data_generator(
    args.train_files,
    args.train_batch_size,
    args.train_max_seq_len,
    grad_accum_steps=grad_accum_steps
)
training_time_ms = 0
validation_time_ms = 0

# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations  # num_scheduled_iterations + num_extension_iterations
ws_short = ws_short_schedule[0]
ws_long = ws_long_schedule[0]
step = 0
while step <= train_steps:
    last_step = (step == train_steps)

    ws_short = ws_short_schedule[step]
    new_ws_long = ws_long_schedule[step]

    if new_ws_long != ws_long:
        model.yarn.apply(ws_long, new_ws_long)
        ws_long=new_ws_long

    # --------------- VALIDATION SECTION -----------------
    # Run validation at regular intervals, at the end of main iterations, and at the final step
    end_of_main_iterations = (step == args.num_iterations)
    if last_step or end_of_main_iterations or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        if last_step:
            ws_long = args.ws_validate_post_yarn_ext  # Extended window size for final validation
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)

        # Start validation timing
        val_t0 = time.perf_counter()

        model.eval()
        assert args.val_tokens % args.val_batch_size == 0

        # <<<< Changed >>>>
        # Set val_accum_steps at the top of the notebook to avoid OOM on a
        # 40GB A100.
        val_steps = val_accum_steps * args.val_tokens // args.val_batch_size
        val_loader = distributed_data_generator(
            args.val_files,
            args.val_batch_size,
            -1,
            grad_accum_steps=val_accum_steps, # <-- Changed
            align_to_bos=False
        )

        val_loss = 0

        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets, cum_seqlens = next(val_loader)
                #val_loss += model(inputs, targets, cum_seqlens, ws_short, ws_long)
                val_loss += model(inputs, cum_seqlens, ws_short, ws_long, targets, inference=False)

        # Calculate validation loss
        val_loss = (val_loss * args.val_ratio) / val_steps

        del val_loader

        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)

        # Store validation loss in history
        val_loss_history[step] = float(val_loss.item())

        # Log to wandb
        wb_log0({
            "val/loss": val_loss.item(),
        }, step=step)

        model.train()

        # Stop validation timing
        torch.cuda.synchronize()
        validation_time_ms += 1000 * (time.perf_counter() - val_t0)

        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process:

            final_metrics = {
                "final/val_loss": float(val_loss.item()),
                "final/val_time_ms": int(validation_time_ms),
                "final/train_time_ms": int(training_time_ms),
                "final/train_time": fmt_elapsed(training_time_ms / 1000),
                "final/steps_total": int(step),
            }

            # Log a single point (optional, helps if you ever filter by metrics, not just summary)
            wandb.log(final_metrics)

            # Ensure they're captured as run-level summary scalars for bar charts
            wandb.run.summary.update(final_metrics)

        # Save checkpoint at final step
        save_checkpoint(step, reason="final")
        # the last step only has the validation loop, so break to avoid training
        break

    # Check for early quit condition
    if args.early_quit is not None and step == args.early_quit:
        print0(f"Early quit triggered at step {step}")
        # Save checkpoint before exiting
        save_checkpoint(step, reason="early_quit")

        print0(f"Training ended early at step {step}/{train_steps}")
        break

    # --------------- TRAINING SECTION -----------------
    for _ in range(grad_accum_steps):
        inputs, targets, cum_seqlens = next(train_loader)
        model(inputs, cum_seqlens, ws_short, ws_long, targets, inference=False).backward()

    # Use new step_optimizers function which handles lr, momentum, and stepping
    step_optimizers(step, optimizers, model)

    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

    # Store timing information in history
    cumulative_train_time_history[step] = approx_training_time_ms
    step_time_history[step] = approx_training_time_ms / (step + 1)  # Average time per step

    # Log to wandb
    idx = min(step, len(adam_lr_schedule) - 1)
    wb_log0({
        "train/short_win":   ws_short_schedule[step] * args.block_size,
        "train/long_win":    ws_long_schedule[step] * args.block_size,
        "train/momentum":    momentum_schedule[idx],
        "train/adam_lr":     adam_lr_schedule[idx],
        "train/muon_lr":     muon_lr_schedule[idx],
        "time/train_ms":    approx_training_time_ms,
        "time/step_avg_ms": approx_training_time_ms / (step + 1),
    }, step=step)


    # Increment step for next iteration
    step += 1

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)

print0(f" Since notebook start: {fmt_elapsed(time.time() - full_notebook_t0)}")
print0(f"Training + validation: {fmt_elapsed(time.time() - train_plus_val_t0)}")

# Calculate durations for this config
dur_kernel_warmup = warmup_time  # Already in seconds
dur_train = training_time_ms / 1000.0  # Convert ms to seconds
dur_val = validation_time_ms / 1000.0  # Convert ms to seconds

if master_process:
    # Log timing breakdowns to wandb (both raw and formatted)
    timing_metrics = {
        # Raw times (in seconds, except for ms which stay in ms)
        "timing/kernel_warmup_sec": dur_kernel_warmup,
        "timing/train_sec": dur_train,
        "timing/val_sec": dur_val,
        # Formatted strings
        "timing/kernel_warmup": fmt_elapsed(dur_kernel_warmup),
        "timing/train": fmt_elapsed(dur_train),
        "timing/val": fmt_elapsed(dur_val),
    }

    # Log to wandb
    wandb.log(timing_metrics)
    wandb.run.summary.update(timing_metrics)

    # Add the log file to the wandb run
    wandb.save(args.log_file)

    # End the run.
    wandb.finish()

if not args.run_hellaswag:
    dist.destroy_process_group()

# endregion training