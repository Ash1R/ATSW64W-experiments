"""Generate the six Colab notebooks from a single source of truth.

Run from the repo root:

    python scripts/build_notebooks.py

This regenerates every notebook in ``notebooks/`` so they stay in sync.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/Ash1R/ATSW64W-experiments.git"


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in src.rstrip("\n").split("\n")],
    }


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in src.rstrip("\n").split("\n")],
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP_CELLS = lambda: [
    md("## Colab setup\n"
       "Verify GPU, mount Drive, clone the repo, install requirements, and create the project tree on Drive."),
    code(
        "# 1. GPU check\n"
        "import torch\n"
        "print('CUDA available:', torch.cuda.is_available())\n"
        "if torch.cuda.is_available():\n"
        "    print('GPU:', torch.cuda.get_device_name(0))"
    ),
    code(
        "# 2. Mount Drive (skipped automatically when not on Colab)\n"
        "try:\n"
        "    from google.colab import drive\n"
        "    drive.mount('/content/drive')\n"
        "    PROJECT_ROOT = '/content/drive/MyDrive/cs4782_patchtst_project'\n"
        "    IN_COLAB = True\n"
        "except Exception:\n"
        "    PROJECT_ROOT = '.'\n"
        "    IN_COLAB = False\n"
        "print('PROJECT_ROOT =', PROJECT_ROOT, ' IN_COLAB =', IN_COLAB)"
    ),
    code(
        "# 3. Clone repo (Colab only). Update REPO_URL in scripts/build_notebooks.py\n"
        "# and re-run that script to regenerate notebooks if the URL changes.\n"
        f"REPO_URL = {REPO_URL!r}\n"
        "if IN_COLAB:\n"
        "    import os, subprocess\n"
        "    os.chdir('/content')\n"
        "    # Derive the clone directory from the URL's basename so it matches the repo name.\n"
        "    REPO_DIRNAME = REPO_URL.rstrip('/').rsplit('/', 1)[-1]\n"
        "    if REPO_DIRNAME.endswith('.git'):\n"
        "        REPO_DIRNAME = REPO_DIRNAME[:-4]\n"
        "    if not os.path.isdir(f'/content/{REPO_DIRNAME}'):\n"
        "        # check=True so a bad URL fails loudly here instead of crashing the next chdir.\n"
        "        subprocess.run(['git', 'clone', REPO_URL, REPO_DIRNAME], check=True)\n"
        "    os.chdir(f'/content/{REPO_DIRNAME}')\n"
        "    subprocess.run(['git', 'pull'], check=False)\n"
        "print('cwd =', __import__('os').getcwd())"
    ),
    code(
        "# 4. Install requirements (best-effort; resolved relative to the repo root\n"
        "# regardless of where the kernel started, so headless `nbconvert` runs work).\n"
        "import subprocess, sys, os\n"
        "_req_dir = os.getcwd()\n"
        "for _ in range(4):\n"
        "    if os.path.isfile(os.path.join(_req_dir, 'requirements.txt')):\n"
        "        break\n"
        "    _req_dir = os.path.dirname(_req_dir)\n"
        "_req = os.path.join(_req_dir, 'requirements.txt')\n"
        "if os.path.isfile(_req):\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', _req], check=False)\n"
        "else:\n"
        "    print('skipping pip install — requirements.txt not found from', os.getcwd())"
    ),
    code(
        "# 5. Make the project's `code/` directory importable.\n"
        "# We add `code/` itself to sys.path (not the repo root) because the\n"
        "# stdlib already ships a module named `code` that the IPython kernel\n"
        "# imports before this cell runs — shadowing that cleanly is messy.\n"
        "# This way every import is `from utils...`, `from data...`, `from models...`.\n"
        "import sys, os\n"
        "# When run with `jupyter nbconvert --execute`, the kernel's cwd is\n"
        "# the notebook's directory (`notebooks/`), so locate the repo root\n"
        "# by walking up until we find `code/`. On Colab we already chdir'd\n"
        "# into the cloned repo above.\n"
        "REPO_DIR = os.getcwd()\n"
        "for _ in range(4):\n"
        "    if os.path.isdir(os.path.join(REPO_DIR, 'code')):\n"
        "        break\n"
        "    REPO_DIR = os.path.dirname(REPO_DIR)\n"
        "CODE_DIR = os.path.join(REPO_DIR, 'code')\n"
        "if not os.path.isdir(CODE_DIR):\n"
        "    raise RuntimeError(f'could not locate code/ from {os.getcwd()}')\n"
        "os.chdir(REPO_DIR)\n"
        "for p in (REPO_DIR, CODE_DIR):\n"
        "    if p not in sys.path:\n"
        "        sys.path.insert(0, p)\n"
        "print('REPO_DIR =', REPO_DIR)\n"
        "from utils.colab import ensure_dirs\n"
        "subdirs = ensure_dirs(PROJECT_ROOT)\n"
        "for k, v in subdirs.items():\n"
        "    print(f'{k:>12}  {v}')"
    ),
]


def nb00():
    cells = [md("# 00 — Colab setup and data verification\n"
                "Run this once per session before the others.")]
    cells.extend(SETUP_CELLS())
    cells += [
        md("## Verify a dataset CSV exists and load one window"),
        code(
            "from data.dataset import build_data_bundle\n"
            "bundle = build_data_bundle(PROJECT_ROOT, 'weather', seq_len=336, pred_len=96)\n"
            "print('weather: num_channels =', bundle.num_channels,\n"
            "      ' train =', len(bundle.train), ' val =', len(bundle.val), ' test =', len(bundle.test))"
        ),
        code(
            "from torch.utils.data import DataLoader\n"
            "loader = DataLoader(bundle.train, batch_size=8, shuffle=False)\n"
            "x, y = next(iter(loader))\n"
            "print('x shape:', tuple(x.shape))   # [B, 336, M]\n"
            "print('y shape:', tuple(y.shape))   # [B, 96, M]"
        ),
        md("If the cell above raised `FileNotFoundError`, follow `data/README.md` and place the CSV under "
           "`<PROJECT_ROOT>/data/weather/weather.csv` on Drive, then re-run."),
        md("## (Optional) Verify Electricity"),
        code(
            "try:\n"
            "    eb = build_data_bundle(PROJECT_ROOT, 'electricity', seq_len=336, pred_len=96)\n"
            "    print('electricity: num_channels =', eb.num_channels,\n"
            "          ' train =', len(eb.train), ' val =', len(eb.val), ' test =', len(eb.test))\n"
            "except FileNotFoundError as e:\n"
            "    print('Electricity not present yet:', e)"
        ),
    ]
    return notebook(cells)


def nb01():
    cells = [md("# 01 — Train baseline (naive)\n"
                "Four runs: {weather, electricity} × {T=96, T=336}. The naive last-value "
                "baseline is the trivial sanity floor before PatchTST.\n\n"
                "DLinear is no longer trained locally — we compare against the DLinear "
                "numbers reported in the PatchTST paper (Table 3) instead. See nb05.")]
    cells.extend(SETUP_CELLS())
    cells += [
        code(
            "from run_experiment import load_config\n"
            "from train import train_from_config\n"
            "import os\n"
            "OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results')\n\n"
            "def run(config_path, **overrides):\n"
            "    cfg = load_config(config_path)\n"
            "    cfg['project_root'] = PROJECT_ROOT\n"
            "    cfg['output_dir'] = OUTPUT_DIR\n"
            "    cfg.update(overrides)\n"
            "    return train_from_config(cfg)"
        ),
        md("## Naive baseline"),
        code(
            "for ds in ['weather', 'electricity']:\n"
            "    for T in [96, 336]:\n"
            "        cfg = {\n"
            "            'model': 'naive', 'dataset': ds,\n"
            "            'seq_len': 336, 'pred_len': T,\n"
            "            'batch_size': 32, 'epochs': 1, 'lr': 1e-4,\n"
            "            'project_root': PROJECT_ROOT, 'output_dir': OUTPUT_DIR,\n"
            "            'seed': 42, 'patch_len': 16, 'stride': 8,\n"
            "        }\n"
            "        try:\n"
            "            train_from_config(cfg)\n"
            "        except FileNotFoundError as e:\n"
            "            print('skip:', e)"
        ),
    ]
    return notebook(cells)


def nb02():
    cells = [md("# 02 — Train PatchTST\n"
                "Cell order: small-debug run, then full Weather/Electricity at T=96 and T=336.\n\n"
                "Resumable: if Colab disconnects, re-run the cell — `resume=True` is on by default.")]
    cells.extend(SETUP_CELLS())
    cells += [
        code(
            "from run_experiment import load_config\n"
            "from train import train_from_config\n"
            "import os\n"
            "OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results')\n\n"
            "def run(config_path, **overrides):\n"
            "    cfg = load_config(config_path)\n"
            "    cfg['project_root'] = PROJECT_ROOT\n"
            "    cfg['output_dir'] = OUTPUT_DIR\n"
            "    cfg['resume'] = True\n"
            "    cfg.update(overrides)\n"
            "    return train_from_config(cfg)"
        ),
        md("### Debug run (1 epoch, small model) — confirms training loop on this Colab GPU"),
        code(
            "run('configs/weather_patchtst_96.yaml',\n"
            "    epochs=1, d_model=64, n_heads=4, num_layers=2, d_ff=128,\n"
            "    run_name='weather_patchtst_T96_DEBUG')"
        ),
        md("### Weather PatchTST T=96"),
        code("run('configs/weather_patchtst_96.yaml')"),
        md("### Weather PatchTST T=336"),
        code("run('configs/weather_patchtst_336.yaml')"),
        md("### Electricity PatchTST T=96"),
        code("run('configs/electricity_patchtst_96.yaml')"),
        md("### Electricity PatchTST T=336"),
        code("run('configs/electricity_patchtst_336.yaml')"),
        md("### (Optional) Traffic T=96 — only if memory permits"),
        code(
            "import torch\n"
            "if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 14e9:\n"
            "    run('configs/electricity_patchtst_96.yaml',\n"
            "        dataset='traffic', batch_size=8,\n"
            "        run_name='traffic_patchtst_T96')\n"
            "else:\n"
            "    print('Skipping Traffic — needs >=16GB GPU and the Traffic CSV.')"
        ),
    ]
    return notebook(cells)


def nb03():
    cells = [md("# 03 — Ablation experiments\n"
                "1. No-patching ablation (Weather T=96)\n"
                "2. Patch-size sweep (P=8/S=4, P=16/S=8, P=32/S=16) on Weather T=96\n"
                "3. Look-back sweep (L=96, 192, 336) on Weather T=96\n"
                "4. Optional: channel-mixing patch Transformer on Weather T=96")]
    cells.extend(SETUP_CELLS())
    cells += [
        code(
            "from run_experiment import load_config\n"
            "from train import train_from_config\n"
            "import os\n"
            "OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results')\n\n"
            "def run_with(base_path, **overrides):\n"
            "    cfg = load_config(base_path)\n"
            "    cfg['project_root'] = PROJECT_ROOT\n"
            "    cfg['output_dir'] = OUTPUT_DIR\n"
            "    cfg.update(overrides)\n"
            "    return train_from_config(cfg)"
        ),
        md("## 1. No-patching ablation"),
        code(
            "run_with('configs/weather_ablation_no_patch_96.yaml',\n"
            "        run_name='weather_patchtst_T96_NOPATCH')"
        ),
        md("## 2. Patch-size sweep (P, S) ∈ {(8,4), (16,8), (32,16)}"),
        code(
            "for P, S in [(8, 4), (16, 8), (32, 16)]:\n"
            "    run_with('configs/weather_ablation_patch_sweep.yaml',\n"
            "            patch_len=P, stride=S,\n"
            "            run_name=f'weather_patchtst_T96_P{P}S{S}')"
        ),
        md("## 3. Look-back sweep L ∈ {96, 192, 336}"),
        code(
            "for L in [96, 192, 336]:\n"
            "    run_with('configs/weather_ablation_lookback_sweep.yaml',\n"
            "            seq_len=L,\n"
            "            run_name=f'weather_patchtst_T96_L{L}')"
        ),
        md("## 4. (Optional) Channel-mixing patch Transformer"),
        code(
            "run_with('configs/weather_ablation_patch_sweep.yaml',\n"
            "        model='channel_mixing', n_heads=8,\n"
            "        run_name='weather_channelmixing_T96')"
        ),
    ]
    return notebook(cells)


def nb04():
    cells = [md("# 04 — Ambitious experiments\n"
                "Run only after notebooks 01–03 produce stable numbers.\n\n"
                "**A.** Runtime/memory profiling — PatchTST P16/S8 vs no-patch ablation, Weather T=96.\n"
                "**B.** Traffic T=96 (resource-permitting) — naive + PatchTST only.\n"
                "**C.** Self-supervised masked patch pretraining on Weather, then fine-tune.\n"
                "**D.** Transfer learning: pretrain on Electricity → fine-tune on Weather T=96.\n"
                "**E.** Failure-case analysis — pick 3 representative test windows.")]
    cells.extend(SETUP_CELLS())
    cells += [
        md("## A. Runtime / memory profiling"),
        code(
            "from run_experiment import load_config\n"
            "from train import train_from_config\n"
            "import os, json\n"
            "OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results')\n\n"
            "# Section A is profiling only — params / per-epoch time / peak memory, not\n"
            "# convergence. The no_patch config (P=1,S=1) has 336 tokens per channel, so\n"
            "# attention is O(N^2) and ~64x slower than patched. 1 epoch is enough to\n"
            "# populate profile_*.json. DLinear was previously profiled here too but is\n"
            "# now compared against via the PatchTST paper baselines (see nb05).\n"
            "specs = [\n"
            "    ('configs/weather_patchtst_96.yaml',          {'run_name': 'profile_patchtst_T96'}),\n"
            "    ('configs/weather_ablation_no_patch_96.yaml', {'run_name': 'profile_nopatch_T96'}),\n"
            "]\n"
            "for path, ov in specs:\n"
            "    cfg = load_config(path)\n"
            "    cfg.update({'project_root': PROJECT_ROOT, 'output_dir': OUTPUT_DIR, 'epochs': 1, 'patience': 1})\n"
            "    cfg.update(ov)\n"
            "    train_from_config(cfg)"
        ),
        code(
            "import glob, json\n"
            "# Skip stale profile_dlinear_*.json from previous runs — DLinear is no\n"
            "# longer profiled locally; we cite the PatchTST paper for its numbers.\n"
            "for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, 'metrics', 'profile_*.json'))):\n"
            "    if 'dlinear' in os.path.basename(f):\n"
            "        continue\n"
            "    s = json.load(open(f))\n"
            "    print(f\"{s['run_name']:>30}  params={s['num_params']:>9,}  \"\n"
            "          f\"avg_epoch_s={s['avg_epoch_seconds']:6.2f}  \"\n"
            "          f\"N={s.get('num_patches','-')}  N^2={s.get('attention_pairs','-')}  \"\n"
            "          f\"peak_mb={s['gpu_memory_mb'].get('peak_alloc_mb','-')}\")"
        ),
        md("## B. Traffic T=96"),
        code(
            "import torch\n"
            "if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 14e9:\n"
            "    for cfg_path, model_overrides in [\n"
            "        ('configs/weather_patchtst_96.yaml', {'model': 'naive', 'dataset': 'traffic', 'run_name': 'traffic_naive_T96'}),\n"
            "        ('configs/weather_patchtst_96.yaml', {'dataset': 'traffic', 'batch_size': 8, 'run_name': 'traffic_patchtst_T96'}),\n"
            "    ]:\n"
            "        cfg = load_config(cfg_path)\n"
            "        cfg.update({'project_root': PROJECT_ROOT, 'output_dir': OUTPUT_DIR})\n"
            "        cfg.update(model_overrides)\n"
            "        train_from_config(cfg)\n"
            "else:\n"
            "    print('Skipping Traffic — needs >=16GB GPU.')"
        ),
        md("## C. Self-supervised masked patch pretraining\n"
           "Mask 40% of non-overlapping patches and reconstruct them with MSE. After pretraining "
           "we save a checkpoint and use `weather_selfsupervised_96.yaml` to fine-tune."),
        code(
            "import torch, numpy as np\n"
            "import torch.nn as nn\n"
            "from torch.utils.data import DataLoader\n"
            "from data.dataset import build_data_bundle\n"
            "from models.patchtst import PatchTST, _num_patches\n\n"
            "SEQ_LEN, PATCH, STRIDE = 336, 16, 16\n"
            "MASK_RATIO = 0.4\n"
            "D_MODEL = 128\n\n"
            "bundle = build_data_bundle(PROJECT_ROOT, 'weather', seq_len=SEQ_LEN, pred_len=96)\n"
            "loader = DataLoader(bundle.train, batch_size=32, shuffle=True, drop_last=True)\n\n"
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
            "ssl_model = PatchTST(seq_len=SEQ_LEN, pred_len=96, patch_len=PATCH, stride=STRIDE,\n"
            "                     d_model=D_MODEL, n_heads=16, num_layers=3, d_ff=256, dropout=0.2).to(device)\n"
            "recon_head = nn.Linear(D_MODEL, PATCH).to(device)\n"
            "opt = torch.optim.Adam(list(ssl_model.parameters()) + list(recon_head.parameters()), lr=1e-4)\n\n"
            "for epoch in range(20):\n"
            "    ssl_model.train()\n"
            "    total = 0.0; n = 0\n"
            "    for x, _ in loader:\n"
            "        x = x.to(device)\n"
            "        x_t = x.transpose(1, 2)\n"
            "        x_norm, mean, std = PatchTST._instance_norm(x_t)\n"
            "        patches = ssl_model.embed(x_norm)\n"
            "        pad = x_norm[..., -1:].expand(-1, -1, STRIDE)\n"
            "        true_patches = torch.cat([x_norm, pad], -1).unfold(-1, PATCH, STRIDE)\n"
            "        B, M, N, _ = patches.shape\n"
            "        mask = (torch.rand(B, M, N, device=device) < MASK_RATIO).float()\n"
            "        masked_patches = patches * (1 - mask).unsqueeze(-1)\n"
            "        tokens = masked_patches.reshape(B * M, N, D_MODEL) + ssl_model.pos_embed\n"
            "        encoded = ssl_model.encoder(ssl_model.encoder_dropout(tokens))\n"
            "        recon = recon_head(encoded).reshape(B, M, N, PATCH)\n"
            "        loss = (((recon - true_patches) ** 2) * mask.unsqueeze(-1)).sum() \\\n"
            "                / mask.sum().clamp(min=1) / PATCH\n"
            "        opt.zero_grad(); loss.backward(); opt.step()\n"
            "        total += loss.item(); n += 1\n"
            "    print(f'[ssl epoch {epoch}] loss={total/max(n,1):.4f}')\n"
            "\n"
            "ckpt_path = os.path.join(OUTPUT_DIR, 'checkpoints', 'weather_patchtst_T96_pretrained_finetune.pt')\n"
            "torch.save({\n"
            "    'model': ssl_model.state_dict(),\n"
            "    'optimizer': None, 'epoch': -1, 'best_val': float('inf'),\n"
            "    'config': {'model': 'patchtst', 'dataset': 'weather', 'seq_len': SEQ_LEN, 'pred_len': 96,\n"
            "               'patch_len': PATCH, 'stride': STRIDE, 'd_model': D_MODEL, 'n_heads': 16,\n"
            "               'num_layers': 3, 'd_ff': 256, 'dropout': 0.2},\n"
            "    'scaler_mean': bundle.scaler.mean, 'scaler_std': bundle.scaler.std,\n"
            "}, ckpt_path)\n"
            "print('saved', ckpt_path)"
        ),
        code(
            "from run_experiment import load_config\n"
            "cfg = load_config('configs/weather_selfsupervised_96.yaml')\n"
            "cfg.update({'project_root': PROJECT_ROOT, 'output_dir': OUTPUT_DIR, 'resume': True,\n"
            "            'patch_len': 16, 'stride': 16})\n"
            "train_from_config(cfg)"
        ),
        md("## D. Transfer learning\n"
           "Caveat: M differs across datasets, so the head must be re-initialised on the target. "
           "Re-run the SSL loop with `dataset='electricity'` and a different ckpt name, then load "
           "only the encoder + patch embedding weights for Weather fine-tuning."),
        md("## E. Failure-case analysis"),
        code(
            "import numpy as np, torch\n"
            "from torch.utils.data import DataLoader\n"
            "from data.dataset import build_data_bundle\n"
            "from models.patchtst import PatchTST\n"
            "\n"
            "ckpt_path = os.path.join(OUTPUT_DIR, 'checkpoints', 'weather_patchtst_L336_T96_P16S8_seed42.pt')\n"
            "# weights_only=False because the checkpoint stores numpy arrays\n"
            "# (scaler.mean/std) alongside the state_dict; PyTorch 2.6 changed\n"
            "# the default to True. Safe here — we wrote the checkpoint ourselves.\n"
            "ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)\n"
            "cfg = ckpt['config']\n"
            "bundle = build_data_bundle(PROJECT_ROOT, cfg['dataset'], cfg['seq_len'], cfg['pred_len'])\n"
            "model = PatchTST(seq_len=cfg['seq_len'], pred_len=cfg['pred_len'], patch_len=cfg['patch_len'],\n"
            "                 stride=cfg['stride'], d_model=cfg['d_model'], n_heads=cfg['n_heads'],\n"
            "                 num_layers=cfg['num_layers'], d_ff=cfg['d_ff'], dropout=cfg['dropout'])\n"
            "model.load_state_dict(ckpt['model']); model.train(False)\n"
            "\n"
            "loader = DataLoader(bundle.test, batch_size=64, shuffle=False)\n"
            "preds, trues, hists = [], [], []\n"
            "with torch.no_grad():\n"
            "    for x, y in loader:\n"
            "        preds.append(model(x).numpy()); trues.append(y.numpy()); hists.append(x.numpy())\n"
            "preds = np.concatenate(preds); trues = np.concatenate(trues); hists = np.concatenate(hists)\n"
            "errs = ((preds - trues) ** 2).mean(axis=(1, 2))\n"
            "good_idx = int(errs.argsort()[len(errs)//20])\n"
            "med_idx = int(errs.argsort()[len(errs)//2])\n"
            "bad_idx = int(errs.argsort()[-1])\n"
            "print('selected indices:', good_idx, med_idx, bad_idx)\n"
            "\n"
            "from utils.plotting import plot_prediction_sample\n"
            "fig_dir = os.path.join(OUTPUT_DIR, 'figures')\n"
            "for tag, idx in [('good', good_idx), ('median', med_idx), ('failure', bad_idx)]:\n"
            "    plot_prediction_sample(\n"
            "        bundle.scaler.inverse(hists[idx]),\n"
            "        bundle.scaler.inverse(trues[idx]),\n"
            "        bundle.scaler.inverse(preds[idx]),\n"
            "        os.path.join(fig_dir, f'failure_case_{tag}.png'),\n"
            "        title=f'Weather T=96 — {tag} (idx={idx})',\n"
            "    )\n"
            "print('wrote failure-case plots to', fig_dir)"
        ),
    ]
    return notebook(cells)


def nb05():
    cells = [md("# 05 — Generate final tables and figures\n"
                "Aggregates every JSON in `results/metrics/` into the main and ablation tables, "
                "then renders the four poster figures.")]
    cells.extend(SETUP_CELLS())
    cells += [
        code(
            "import os, glob, json\n"
            "import numpy as np, pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results')\n"
            "TABLES = os.path.join(OUTPUT_DIR, 'tables'); os.makedirs(TABLES, exist_ok=True)\n"
            "FIGURES = os.path.join(OUTPUT_DIR, 'figures'); os.makedirs(FIGURES, exist_ok=True)\n\n"
            "# DLinear is no longer trained locally — we cite the PatchTST paper (Nie et al.,\n"
            "# ICLR 2023) Table 3 instead. TODO: verify values against your copy of Table 3\n"
            "# (multivariate, normalized test MSE, look-back 336).\n"
            "PAPER_DLINEAR_MSE_NORM = {\n"
            "    ('weather',     96):  0.196,\n"
            "    ('weather',    336):  0.283,\n"
            "    ('electricity', 96):  0.140,\n"
            "    ('electricity',336):  0.169,\n"
            "    ('traffic',     96):  0.410,\n"
            "    ('traffic',    336):  0.436,\n"
            "}\n\n"
            "rows = []\n"
            "for path in sorted(glob.glob(os.path.join(OUTPUT_DIR, 'metrics', '*.json'))):\n"
            "    s = json.load(open(path))\n"
            "    cfg = s.get('config', {})\n"
            "    if cfg.get('model') == 'dlinear':\n"
            "        continue   # stale local DLinear runs no longer used; see PAPER_DLINEAR_MSE_NORM\n"
            "    rows.append({\n"
            "        'run_name': s['run_name'],\n"
            "        'dataset': cfg.get('dataset'),\n"
            "        'model': cfg.get('model'),\n"
            "        'seq_len': cfg.get('seq_len'),\n"
            "        'pred_len': cfg.get('pred_len'),\n"
            "        'patch_len': cfg.get('patch_len'),\n"
            "        'stride': cfg.get('stride'),\n"
            "        'mse_inv': s.get('test_mse_inverse'),\n"
            "        'mae_inv': s.get('test_mae_inverse'),\n"
            "        'mse_norm': s.get('test_mse_normalized'),\n"
            "        'mae_norm': s.get('test_mae_normalized'),\n"
            "        'num_patches': s.get('num_patches'),\n"
            "        'attention_pairs': s.get('attention_pairs'),\n"
            "        'avg_epoch_s': s.get('avg_epoch_seconds'),\n"
            "        'num_params': s.get('num_params'),\n"
            "    })\n"
            "df = pd.DataFrame(rows)\n"
            "df.to_csv(os.path.join(TABLES, 'all_runs.csv'), index=False)\n"
            "df"
        ),
        md("## Main results table — naive vs PatchTST (DLinear baseline cited from paper)"),
        code(
            "PAPER_PATCHTST_MSE_NORM = {('weather', 96): 0.152, ('weather', 336): 0.249,\n"
            "                           ('electricity', 96): 0.130, ('electricity', 336): 0.167}\n\n"
            "# Use NORMALIZED MSE for the cross-dataset main table — the paper's metric.\n"
            "# `mse_inv` is in raw physical units and is not comparable between datasets.\n"
            "main = (df[df['model'].isin(['naive', 'patchtst']) &\n"
            "          (df['patch_len'].fillna(16) == 16) & (df['stride'].fillna(8) == 8)]\n"
            "        .pivot_table(index=['dataset', 'pred_len'], columns='model',\n"
            "                     values='mse_norm', aggfunc='min')\n"
            "        .reset_index())\n"
            "main['paper_dlinear']  = main.apply(lambda r: PAPER_DLINEAR_MSE_NORM.get((r['dataset'], r['pred_len'])), axis=1)\n"
            "main['paper_patchtst'] = main.apply(lambda r: PAPER_PATCHTST_MSE_NORM.get((r['dataset'], r['pred_len'])), axis=1)\n"
            "main = main[['dataset', 'pred_len', 'naive', 'paper_dlinear', 'patchtst', 'paper_patchtst']]\n"
            "main.to_csv(os.path.join(TABLES, 'main_results.csv'), index=False)\n"
            "with open(os.path.join(TABLES, 'main_results_latex.txt'), 'w') as f:\n"
            "    f.write(main.to_latex(index=False, float_format='%.4f'))\n"
            "main"
        ),
        md("## Ablation table"),
        code(
            "abl_rows = df[(df['dataset'] == 'weather') & (df['pred_len'] == 96)\n"
            "              & df['model'].isin(['patchtst'])\n"
            "              & df['run_name'].str.contains('NOPATCH|P\\\\d+S\\\\d+|T96_L\\\\d+$', regex=True)]\n"
            "abl = abl_rows[['run_name', 'patch_len', 'stride', 'seq_len', 'num_patches',\n"
            "                'attention_pairs', 'mse_inv', 'mae_inv', 'avg_epoch_s']].copy()\n"
            "abl.to_csv(os.path.join(TABLES, 'ablation_results.csv'), index=False)\n"
            "with open(os.path.join(TABLES, 'ablation_results_latex.txt'), 'w') as f:\n"
            "    f.write(abl.to_latex(index=False, float_format='%.4f'))\n"
            "abl"
        ),
        md("## Figure 1 — main results bar chart"),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)\n"
            "for ax, T in zip(axes, [96, 336]):\n"
            "    sub = main[main['pred_len'] == T].set_index('dataset')\n"
            "    sub[['naive', 'paper_dlinear', 'patchtst']].plot.bar(ax=ax)\n"
            "    ax.set_title(f'Test MSE @ T={T}  (DLinear from PatchTST paper Table 3)')\n"
            "    ax.set_ylabel('MSE (normalized)')\n"
            "    ax.tick_params(axis='x', rotation=0)\n"
            "fig.tight_layout()\n"
            "fig.savefig(os.path.join(FIGURES, 'main_results_bar.png'), dpi=150)\n"
            "plt.show()"
        ),
        md("## Figure 2 — patching efficiency (token count vs MSE)"),
        code(
            "eff = abl.dropna(subset=['num_patches']).copy()\n"
            "fig, ax = plt.subplots(figsize=(6, 4))\n"
            "ax.scatter(eff['num_patches'], eff['mse_inv'], s=80)\n"
            "for _, r in eff.iterrows():\n"
            "    ax.annotate(f\"P={int(r['patch_len'])}/S={int(r['stride'])}\",\n"
            "                (r['num_patches'], r['mse_inv']),\n"
            "                textcoords='offset points', xytext=(6, 4))\n"
            "ax.set_xscale('log')\n"
            "ax.set_xlabel('# tokens N (log)')\n"
            "ax.set_ylabel('Test MSE (inverse-scaled)')\n"
            "ax.set_title('Patching efficiency on Weather T=96')\n"
            "ax.grid(True, alpha=0.3)\n"
            "fig.tight_layout()\n"
            "fig.savefig(os.path.join(FIGURES, 'patching_efficiency.png'), dpi=150)\n"
            "plt.show()"
        ),
        md("## Figure 3 — patch-size and look-back sweeps"),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
            "psweep = abl[abl['run_name'].str.contains('P\\\\d+S\\\\d+', regex=True)].sort_values('patch_len')\n"
            "axes[0].plot(psweep['patch_len'], psweep['mse_inv'], marker='o')\n"
            "axes[0].set_xlabel('patch length P'); axes[0].set_ylabel('MSE'); axes[0].set_title('Patch-size sweep')\n"
            "axes[0].grid(True, alpha=0.3)\n"
            "lsweep = abl[abl['run_name'].str.contains('T96_L\\\\d+$', regex=True)].sort_values('seq_len')\n"
            "axes[1].plot(lsweep['seq_len'], lsweep['mse_inv'], marker='s', color='tab:orange')\n"
            "axes[1].set_xlabel('look-back length L'); axes[1].set_ylabel('MSE'); axes[1].set_title('Look-back sweep')\n"
            "axes[1].grid(True, alpha=0.3)\n"
            "fig.tight_layout()\n"
            "fig.savefig(os.path.join(FIGURES, 'patch_size_sweep.png'), dpi=150)\n"
            "fig.savefig(os.path.join(FIGURES, 'lookback_sweep.png'), dpi=150)\n"
            "plt.show()"
        ),
        md("## Figure 4 — example prediction"),
        code(
            "import shutil\n"
            "src = os.path.join(FIGURES, 'weather_patchtst_L336_T96_P16S8_seed42_prediction_plot.png')\n"
            "dst = os.path.join(FIGURES, 'example_predictions.png')\n"
            "if os.path.exists(src):\n"
            "    shutil.copyfile(src, dst)\n"
            "    print('copied to', dst)\n"
            "else:\n"
            "    print('expected', src, 'not found — run notebook 02 first')"
        ),
    ]
    return notebook(cells)


def nb06():
    """Beyond-the-paper extensions. Two experiments not in PatchTST (Nie 2023).

    Section A - setup + load checkpoint
    Section B - patch-shuffle robustness curve              [inference only, ~1 min]
    Section C - channel-grouping spectrum (k = 1, 3, 7, 21) [trains 4 models, ~60-90 min total]
    Section D - render both poster figures inline
    Section E - copy figures to project_root/results/figures and print summary

    Run all cells top-to-bottom and you're done — no separate scripts.
    """
    cells = SETUP_CELLS() + [
        md("## Beyond-the-paper extensions\n"
           "Two experiments not in the PatchTST paper:\n\n"
           "  1. **Patch order shuffling** (inference-only, ~1 min)\n"
           "  2. **Channel grouping spectrum** (trains 4 models, ~60-90 min on a T4)\n\n"
           "**This notebook is fully self-contained** — both experiment implementations are "
           "defined inline below, so it works whether or not the GitHub clone has the latest "
           "`code/extensions/` files. Just run all cells top-to-bottom. The notebook writes the "
           "raw extension JSON files and a complete `nb06_extension_results_manifest.json` to "
           "`<project_root>/results/extensions/`, then renders the two final figures inline and "
           "saves them to `<project_root>/results/figures/`. On Colab, `<project_root>` is on "
           "Google Drive, so these outputs persist after the runtime stops."),

        # --- Section A: setup + load checkpoint ---
        md("### A. Setup and load the headline PatchTST checkpoint\n"
           "Imports, paths, and the loaded `weather_patchtst_L336_T96_P16S8_seed42` model."),
        code(
            "# Imports — use top-level package names because SETUP_CELLS added\n"
            "# `code/` directly to sys.path (the stdlib already owns `code`).\n"
            "import os, json\n"
            "import numpy as np, torch\n"
            "import torch.nn as nn\n"
            "import matplotlib.pyplot as plt\n"
            "from data.dataset import build_data_bundle\n"
            "from data.preprocessing import StandardScaler\n"
            "from models import build_model\n"
            "from models.patchtst import _num_patches\n"
            "from utils.device import get_device\n"
            "from train import train_from_config\n"
            "\n"
            "EXT_DIR = os.path.join(PROJECT_ROOT, 'results', 'extensions')\n"
            "FIG_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures')\n"
            "os.makedirs(EXT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)\n"
            "device = get_device()\n"
            "print('device:', device)"
        ),
        code(
            "CKPT_NAME = 'weather_patchtst_L336_T96_P16S8_seed42'\n"
            "ckpt_path = os.path.join(PROJECT_ROOT, 'results', 'checkpoints', CKPT_NAME + '.pt')\n"
            "ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)\n"
            "cfg = ckpt['config']\n"
            "bundle = build_data_bundle(\n"
            "    project_root=PROJECT_ROOT, dataset=cfg['dataset'],\n"
            "    seq_len=cfg['seq_len'], pred_len=cfg['pred_len']\n"
            ")\n"
            "bundle.scaler = StandardScaler(\n"
            "    mean=np.asarray(ckpt['scaler_mean']), std=np.asarray(ckpt['scaler_std'])\n"
            ")\n"
            "_, _, test_loader = bundle.loaders(batch_size=64, shuffle_train=False)\n"
            "model = build_model(cfg['model'],\n"
            "    seq_len=cfg['seq_len'], pred_len=cfg['pred_len'], num_channels=bundle.num_channels,\n"
            "    patch_len=cfg.get('patch_len', 16), stride=cfg.get('stride', 8),\n"
            "    d_model=cfg.get('d_model', 128), n_heads=cfg.get('n_heads', 16),\n"
            "    num_layers=cfg.get('num_layers', 3), d_ff=cfg.get('d_ff', 256),\n"
            "    dropout=cfg.get('dropout', 0.2)\n"
            ").to(device)\n"
            "model.load_state_dict(ckpt['model'])\n"
            "model.eval()\n"
            "print(f'loaded {CKPT_NAME} ({sum(p.numel() for p in model.parameters()):,} params)')"
        ),

        # --- Section B: patch shuffle (impl inlined) ---
        md("### B. Patch-content shuffling before positional embeddings\n"
           "Permute a random fraction of patch **contents** at inference; sweep frac in "
           "{0, 0.1, 0.25, 0.5, 0.75, 1.0}.\n\n"
           "**Beyond the paper:** PatchTST never tests whether patch contents must appear in the "
           "correct temporal slots. We shuffle patch contents **before** adding positional "
           "embeddings, so positional slots stay fixed while the wrong local subseries appears "
           "at those slots. If MSE rises, PatchTST depends on temporal patch placement, not just "
           "unordered local patch statistics."),
        code(
            "def _forward_with_pre_pos_content_shuffle(model, x, frac, rng):\n"
            "    B, L, M = x.shape\n"
            "    x_t = x.transpose(1, 2)\n"
            "    x_norm, mean, std = model._instance_norm(x_t)\n"
            "    patches = model.embed(x_norm)  # [B, M, N, d_model]\n"
            "    N = patches.shape[2]\n"
            "    if frac > 0:\n"
            "        n_shuf = max(2, int(round(frac * N)))\n"
            "        idx = np.arange(N)\n"
            "        pos = np.sort(rng.choice(N, size=n_shuf, replace=False))\n"
            "        perm = idx.copy()\n"
            "        perm[pos] = rng.permutation(perm[pos])\n"
            "        perm_t = torch.as_tensor(perm, device=patches.device, dtype=torch.long)\n"
            "        patches = patches.index_select(2, perm_t)\n"
            "    tokens = patches.reshape(B * M, N, model.d_model)\n"
            "    tokens = tokens + model.pos_embed  # positional slots stay fixed\n"
            "    tokens = model.encoder_dropout(tokens)\n"
            "    encoded = model.encoder(tokens)\n"
            "    flat = encoded.reshape(B * M, N * model.d_model)\n"
            "    pred = model.head(flat).reshape(B, M, model.pred_len)\n"
            "    pred = pred * std + mean\n"
            "    return pred.transpose(1, 2)\n"
            "\n"
            "def patch_shuffle_curve(model, loader, device, fractions, seed=0):\n"
            "    N = model.num_patches\n"
            "    out = {'type': 'pre_positional_patch_content_shuffle',\n"
            "           'fractions': list(fractions), 'mse': [], 'mae': []}\n"
            "    model.eval()\n"
            "    for frac in out['fractions']:\n"
            "        rng = np.random.RandomState(seed + int(frac * 1000))\n"
            "        preds, trues = [], []\n"
            "        with torch.no_grad():\n"
            "            for x, y in loader:\n"
            "                preds.append(_forward_with_pre_pos_content_shuffle(\n"
            "                    model, x.to(device), frac, rng).cpu().numpy())\n"
            "                trues.append(y.numpy())\n"
            "        p = np.concatenate(preds, 0); t = np.concatenate(trues, 0)\n"
            "        out['mse'].append(float(((p - t) ** 2).mean()))\n"
            "        out['mae'].append(float(np.abs(p - t).mean()))\n"
            "    return out\n"
            "\n"
            "ps_out = patch_shuffle_curve(model, test_loader, device,\n"
            "    fractions=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0), seed=0)\n"
            "ps_out['baseline_mse'] = ps_out['mse'][0]\n"
            "ps_out['checkpoint'] = CKPT_NAME\n"
            "with open(os.path.join(EXT_DIR, 'patch_shuffle.json'), 'w') as f:\n"
            "    json.dump(ps_out, f, indent=2)\n"
            "for f, m in zip(ps_out['fractions'], ps_out['mse']):\n"
            "    print(f'  shuffled {f*100:5.1f}%  MSE = {m:.4f}')"
        ),

        md("### B2. No positional embedding ablation\n"
           "Evaluate the same trained checkpoint with `pos_embed` zeroed at inference time. "
           "Patch contents remain in the correct order, so this isolates the contribution of "
           "the learned positional embedding itself."),
        code(
            "def evaluate_no_positional_embedding(model, loader, device):\n"
            "    model.eval()\n"
            "    saved_pos = model.pos_embed.detach().clone()\n"
            "    preds, trues = [], []\n"
            "    try:\n"
            "        model.pos_embed.data.zero_()\n"
            "        with torch.no_grad():\n"
            "            for x, y in loader:\n"
            "                preds.append(model(x.to(device)).cpu().numpy())\n"
            "                trues.append(y.numpy())\n"
            "    finally:\n"
            "        model.pos_embed.data.copy_(saved_pos)\n"
            "    p = np.concatenate(preds, 0); t = np.concatenate(trues, 0)\n"
            "    return {'type': 'zero_positional_embedding',\n"
            "            'mse': float(((p - t) ** 2).mean()),\n"
            "            'mae': float(np.abs(p - t).mean())}\n"
            "\n"
            "nopos_out = evaluate_no_positional_embedding(model, test_loader, device)\n"
            "nopos_out['baseline_mse'] = ps_out['baseline_mse']\n"
            "nopos_out['checkpoint'] = CKPT_NAME\n"
            "with open(os.path.join(EXT_DIR, 'no_positional_embedding.json'), 'w') as f:\n"
            "    json.dump(nopos_out, f, indent=2)\n"
            "pct = (nopos_out['mse'] - nopos_out['baseline_mse']) / nopos_out['baseline_mse'] * 100\n"
            "print(f\"no positional embedding  MSE = {nopos_out['mse']:.4f}  ({pct:+.1f}% vs baseline)\")"
        ),

        # --- Section C: channel grouping (model + clustering inlined; trained without going through build_model) ---
        md("### C. Channel grouping spectrum (k = 1, 3, 7, 21)\n"
           "The paper presents the channel-axis treatment as binary (full-mix vs full-indep). "
           "We cluster Weather's 21 channels by Pearson correlation on the train split, then "
           "train one model per k. k=1 reduces to channel-mixing, k=21 to channel-independence.\n\n"
           "**Beyond the paper:** if structured cross-channel mixing recovers signal that pure "
           "independence loses, that's a meaningful new design point. If the spectrum is monotone, "
           "the binary choice in the paper is empirically correct.\n\n"
           "Implementation: model class + clustering helper are defined inline. We train each k "
           "with a manual training loop (Adam, lr 1e-4, 20 epochs, early-stop on val MSE) so the "
           "notebook doesn't need a model-registry change in `code/`."),
        code(
            "# 1. Channel-grouped PatchTST model (defined inline; no code/ changes required)\n"
            "class ChannelGroupedPatchTST(nn.Module):\n"
            "    def __init__(self, seq_len, pred_len, groups,\n"
            "                 patch_len=16, stride=8, d_model=128, n_heads=16,\n"
            "                 num_layers=3, d_ff=256, dropout=0.2):\n"
            "        super().__init__()\n"
            "        self.seq_len, self.pred_len = seq_len, pred_len\n"
            "        self.patch_len, self.stride, self.d_model = patch_len, stride, d_model\n"
            "        self.groups = [sorted(g) for g in groups]\n"
            "        self.num_patches = _num_patches(seq_len, patch_len, stride)\n"
            "        self.group_embed = nn.ModuleList(\n"
            "            nn.Linear(patch_len * len(g), d_model) for g in self.groups)\n"
            "        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))\n"
            "        nn.init.trunc_normal_(self.pos_embed, std=0.02)\n"
            "        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,\n"
            "            dim_feedforward=d_ff, dropout=dropout, activation='gelu',\n"
            "            batch_first=True, norm_first=True)\n"
            "        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)\n"
            "        self.encoder_dropout = nn.Dropout(dropout)\n"
            "        self.group_head = nn.ModuleList(\n"
            "            nn.Linear(self.num_patches * d_model, pred_len * len(g)) for g in self.groups)\n"
            "    @staticmethod\n"
            "    def _instance_norm(x, eps=1e-5):\n"
            "        mean = x.mean(-1, keepdim=True)\n"
            "        std  = x.std(-1, keepdim=True, unbiased=False) + eps\n"
            "        return (x - mean) / std, mean, std\n"
            "    def _patchify(self, x_g):\n"
            "        B, mg, L = x_g.shape\n"
            "        pad = x_g[..., -1:].expand(B, mg, self.stride)\n"
            "        xp = torch.cat([x_g, pad], dim=-1)\n"
            "        p = xp.unfold(-1, self.patch_len, self.stride)\n"
            "        return p.permute(0, 2, 1, 3).reshape(B, self.num_patches, mg * self.patch_len)\n"
            "    def forward(self, x):\n"
            "        B, L, M = x.shape\n"
            "        x = x.transpose(1, 2)\n"
            "        x_norm, mean, std = self._instance_norm(x)\n"
            "        out = torch.zeros(B, M, self.pred_len, device=x.device, dtype=x.dtype)\n"
            "        for g, embed, head in zip(self.groups, self.group_embed, self.group_head):\n"
            "            idx = torch.as_tensor(g, device=x.device, dtype=torch.long)\n"
            "            xg = x_norm.index_select(1, idx)\n"
            "            patches = self._patchify(xg)\n"
            "            tokens = embed(patches) + self.pos_embed\n"
            "            tokens = self.encoder_dropout(tokens)\n"
            "            enc = self.encoder(tokens)\n"
            "            flat = enc.reshape(B, self.num_patches * self.d_model)\n"
            "            pred_g = head(flat).reshape(B, len(g), self.pred_len)\n"
            "            out.index_copy_(1, idx, pred_g)\n"
            "        return (out * std + mean).transpose(1, 2)\n"
            "\n"
            "def cluster_channels_by_correlation(train_data, k, seed=0):\n"
            "    M = train_data.shape[1]\n"
            "    if k <= 1: return [list(range(M))]\n"
            "    if k >= M: return [[i] for i in range(M)]\n"
            "    corr = np.corrcoef(train_data.T); np.fill_diagonal(corr, 0.0)\n"
            "    score = np.abs(corr).mean(1)\n"
            "    order = np.argsort(-score)\n"
            "    rng = np.random.RandomState(seed); rng.shuffle(order)\n"
            "    groups = [[] for _ in range(k)]\n"
            "    for i, ch in enumerate(order):\n"
            "        groups[i % k].append(int(ch))\n"
            "    return [sorted(g) for g in groups]"
        ),
        code(
            "# 2. Manual train loop — same recipe as the rest of the project (Adam, lr 1e-4,\n"
            "#    20 epochs, early-stop on val MSE). Doesn't depend on the build_model registry.\n"
            "from utils.metrics import compute_metrics\n"
            "\n"
            "def train_grouped_model(groups, k, epochs=20, lr=1e-4, patience=5):\n"
            "    torch.manual_seed(42); np.random.seed(42)\n"
            "    model_g = ChannelGroupedPatchTST(\n"
            "        seq_len=cfg['seq_len'], pred_len=cfg['pred_len'], groups=groups,\n"
            "        patch_len=cfg.get('patch_len', 16), stride=cfg.get('stride', 8),\n"
            "        d_model=cfg.get('d_model', 128), n_heads=cfg.get('n_heads', 16),\n"
            "        num_layers=cfg.get('num_layers', 3), d_ff=cfg.get('d_ff', 256),\n"
            "        dropout=cfg.get('dropout', 0.2),\n"
            "    ).to(device)\n"
            "    opt = torch.optim.Adam(model_g.parameters(), lr=lr)\n"
            "    train_loader, val_loader, _ = bundle.loaders(batch_size=32)\n"
            "    best_val, best_state, bad = float('inf'), None, 0\n"
            "    for ep in range(epochs):\n"
            "        model_g.train()\n"
            "        for x, y in train_loader:\n"
            "            x, y = x.to(device), y.to(device)\n"
            "            opt.zero_grad(); loss = ((model_g(x) - y) ** 2).mean()\n"
            "            loss.backward(); opt.step()\n"
            "        model_g.eval()\n"
            "        vals = []\n"
            "        with torch.no_grad():\n"
            "            for x, y in val_loader:\n"
            "                vals.append(((model_g(x.to(device)) - y.to(device)) ** 2).mean().item())\n"
            "        v = float(np.mean(vals))\n"
            "        print(f'  k={k:>2} ep {ep+1:>2}/{epochs}  val_mse={v:.4f}')\n"
            "        if v < best_val:\n"
            "            best_val, best_state, bad = v, {kk: vv.detach().clone() for kk, vv in model_g.state_dict().items()}, 0\n"
            "        else:\n"
            "            bad += 1\n"
            "            if bad >= patience:\n"
            "                print(f'  early-stop at epoch {ep+1}'); break\n"
            "    model_g.load_state_dict(best_state)\n"
            "    # test MSE on normalized scale\n"
            "    model_g.eval(); preds, trues = [], []\n"
            "    with torch.no_grad():\n"
            "        for x, y in test_loader:\n"
            "            preds.append(model_g(x.to(device)).cpu().numpy()); trues.append(y.numpy())\n"
            "    p = np.concatenate(preds, 0); t = np.concatenate(trues, 0)\n"
            "    return {'mse': float(((p - t) ** 2).mean()),\n"
            "            'mae': float(np.abs(p - t).mean()),\n"
            "            'params': sum(pp.numel() for pp in model_g.parameters())}"
        ),
        code(
            "train_data = bundle.train.data\n"
            "cg_results = []\n"
            "cg_groups = {}\n"
            "for k in (1, 3, 7, 21):\n"
            "    groups = cluster_channels_by_correlation(train_data, k=k, seed=0)\n"
            "    cg_groups[str(k)] = groups\n"
            "    print(f'\\nk={k}: {len(groups)} groups, sizes = {[len(g) for g in groups]}')\n"
            "    res = train_grouped_model(groups, k=k, epochs=20, lr=1e-4, patience=5)\n"
            "    cg_results.append({'k': k, **res})\n"
            "    print(f\"  -> test MSE = {res['mse']:.4f}  params = {res['params']:,}\")\n"
            "cg_out = {'k_groups': [r['k'] for r in cg_results],\n"
            "          'mse':      [r['mse'] for r in cg_results],\n"
            "          'mae':      [r['mae'] for r in cg_results],\n"
            "          'params':   [r['params'] for r in cg_results],\n"
            "          'groups':   cg_groups}\n"
            "with open(os.path.join(EXT_DIR, 'channel_grouping.json'), 'w') as f:\n"
            "    json.dump(cg_out, f, indent=2)"
        ),

        # --- Section D: render both figures inline ---
        md("### D. Render the two poster figures\n"
           "Both figures are saved to `<project_root>/results/figures/` and shown inline."),
        code(
            "# Figure 1: pre-positional patch-content shuffle curve\n"
            "fracs = np.array(ps_out['fractions'])\n"
            "mse = np.array(ps_out['mse'])\n"
            "fig, ax = plt.subplots(figsize=(4.4, 2.8), dpi=160)\n"
            "ax.plot(fracs * 100, mse, 'o-', color='#b31b1b', lw=2.2, ms=7)\n"
            "ax.axhline(mse[0], ls='--', color='#777', lw=1, label=f'unshuffled = {mse[0]:.3f}')\n"
            "if 'nopos_out' in globals():\n"
            "    ax.axhline(nopos_out['mse'], ls=':', color='#1f4fa6', lw=1.8,\n"
            "               label=f'no pos embed = {nopos_out[\"mse\"]:.3f}')\n"
            "ax.set_xlabel('Patch contents shuffled before pos embed (%)')\n"
            "ax.set_ylabel('Test MSE (normalized)')\n"
            "ax.set_title('Does temporal patch placement matter?', fontweight='bold')\n"
            "ax.grid(alpha=0.3); ax.legend(loc='upper left', fontsize=9)\n"
            "fig.tight_layout()\n"
            "out_png = os.path.join(FIG_DIR, 'ext_patch_shuffle.png')\n"
            "fig.savefig(out_png, bbox_inches='tight', facecolor='white'); plt.show()\n"
            "print('saved', out_png)"
        ),
        code(
            "# Figure 2: channel grouping spectrum (MSE vs k, with params on a twin axis)\n"
            "ks = np.array(cg_out['k_groups'])\n"
            "mses = np.array(cg_out['mse'])\n"
            "params = np.array([p if p is not None else np.nan for p in cg_out['params']])\n"
            "fig, ax1 = plt.subplots(figsize=(4.6, 2.8), dpi=160)\n"
            "c_red, c_blue = '#b31b1b', '#1f4fa6'\n"
            "ax1.plot(ks, mses, 'o-', color=c_red, lw=2.2, ms=8, label='Test MSE')\n"
            "for k, m in zip(ks, mses):\n"
            "    ax1.annotate(f'{m:.3f}', xy=(k, m), xytext=(0, 8),\n"
            "                 textcoords='offset points', ha='center', fontsize=9, color=c_red)\n"
            "ax1.set_xlabel('Channel groups k  (1 = full mixing  ->  21 = full independence)')\n"
            "ax1.set_ylabel('Test MSE (normalized)', color=c_red)\n"
            "ax1.tick_params(axis='y', labelcolor=c_red)\n"
            "ax1.set_xscale('log'); ax1.set_xticks(ks); ax1.set_xticklabels([str(k) for k in ks])\n"
            "ax1.grid(alpha=0.3)\n"
            "ax2 = ax1.twinx()\n"
            "ax2.plot(ks, params / 1e6, 's--', color=c_blue, lw=1.5, ms=6)\n"
            "ax2.set_ylabel('Parameters (M)', color=c_blue)\n"
            "ax2.tick_params(axis='y', labelcolor=c_blue)\n"
            "ax1.set_title('Mixing  <->  independence spectrum', fontweight='bold')\n"
            "fig.tight_layout()\n"
            "out_png = os.path.join(FIG_DIR, 'ext_channel_grouping.png')\n"
            "fig.savefig(out_png, bbox_inches='tight', facecolor='white'); plt.show()\n"
            "print('saved', out_png)"
        ),

        # --- Section E: summary ---
        md("### E. Push the complete nb6 results payload to Drive\n"
           "This cell writes a single manifest JSON containing the actual extension numbers, "
           "channel groups, figure paths, and source checkpoint. In Colab this lands directly "
           "in Google Drive at `<project_root>/results/extensions/`."),
        code(
            "import datetime as _datetime_module\n"
            "\n"
            "manifest = {\n"
            "    'created_utc': _datetime_module.datetime.now(_datetime_module.timezone.utc).isoformat(),\n"
            "    'project_root': os.path.abspath(PROJECT_ROOT),\n"
            "    'in_colab': bool(IN_COLAB),\n"
            "    'checkpoint': CKPT_NAME,\n"
            "    'checkpoint_path': ckpt_path,\n"
            "    'raw_result_files': {\n"
            "        'patch_shuffle': os.path.join(EXT_DIR, 'patch_shuffle.json'),\n"
            "        'channel_grouping': os.path.join(EXT_DIR, 'channel_grouping.json'),\n"
            "        'no_positional_embedding': os.path.join(EXT_DIR, 'no_positional_embedding.json'),\n"
            "    },\n"
            "    'figure_files': {\n"
            "        'patch_shuffle': os.path.join(FIG_DIR, 'ext_patch_shuffle.png'),\n"
            "        'channel_grouping': os.path.join(FIG_DIR, 'ext_channel_grouping.png'),\n"
            "    },\n"
            "    'patch_shuffle': ps_out,\n"
            "    'no_positional_embedding': nopos_out,\n"
            "    'channel_grouping': cg_out,\n"
            "}\n"
            "manifest_path = os.path.join(EXT_DIR, 'nb06_extension_results_manifest.json')\n"
            "with open(manifest_path, 'w') as f:\n"
            "    json.dump(manifest, f, indent=2)\n"
            "\n"
            "# Best-effort durability hint for Drive-backed paths before the runtime exits.\n"
            "try:\n"
            "    for path in [manifest_path, *manifest['raw_result_files'].values(), *manifest['figure_files'].values()]:\n"
            "        if os.path.exists(path):\n"
            "            with open(path, 'ab') as _f:\n"
            "                _f.flush(); os.fsync(_f.fileno())\n"
            "except Exception as e:\n"
            "    print('Drive fsync skipped:', e)\n"
            "\n"
            "print('nb6 results manifest saved to:', manifest_path)\n"
            "for label, path in manifest['raw_result_files'].items():\n"
            "    print(f'raw {label}:', path)\n"
            "for label, path in manifest['figure_files'].items():\n"
            "    print(f'figure {label}:', path)"
        ),
        md("### F. Summary\n"
           "Both figures are now in `<project_root>/results/figures/` and the underlying numbers "
           "plus the complete nb6 manifest are in `<project_root>/results/extensions/`. Edit the "
           "poster's two extension-panel <img> tags to point at `ext_patch_shuffle.png` and "
           "`ext_channel_grouping.png`."),
        code(
            "print('\\n=== Patch shuffling ===')\n"
            "for f, m in zip(ps_out['fractions'], ps_out['mse']):\n"
            "    pct = (m - ps_out['baseline_mse']) / ps_out['baseline_mse'] * 100\n"
            "    print(f'  shuffled {f*100:5.1f}%  MSE = {m:.4f}  ({pct:+.1f}% vs unshuffled)')\n"
            "print('\\n=== Channel grouping ===')\n"
            "for k, m, p in zip(cg_out['k_groups'], cg_out['mse'], cg_out['params']):\n"
            "    label = ('full mixing' if k == 1 else 'full independence' if k == 21 else f'k={k}')\n"
            "    print(f'  k={k:>2} ({label:<18}) MSE = {m:.4f}  params = {p:,}')\n"
            "print('\\nManifest:', manifest_path)\n"
            "print('\\nDone. Re-export the poster PDF to pick up the new figures.')"
        ),
    ]
    return notebook(cells)


def main():
    out_dir = Path("notebooks")
    out_dir.mkdir(exist_ok=True)
    builders = {
        "00_colab_setup_and_data.ipynb": nb00,
        "01_train_baselines_colab.ipynb": nb01,
        "02_train_patchtst_colab.ipynb": nb02,
        "03_ablation_experiments_colab.ipynb": nb03,
        "04_ambitious_experiments_colab.ipynb": nb04,
        "05_generate_results_figures_colab.ipynb": nb05,
        "06_extension_experiments_colab.ipynb": nb06,
    }
    for name, fn in builders.items():
        path = out_dir / name
        path.write_text(json.dumps(fn(), indent=1))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
