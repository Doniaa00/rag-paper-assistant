# Environment Notes

Machine-specific quirks worth knowing before debugging "it doesn't work here"
as a code problem.

## Python 3.14

This environment runs Python 3.14, which is unusually new (released Oct
2025). Some packages lag behind on wheel availability or compatibility for
brand-new Python versions. If a future `pip install` fails, behaves oddly, or
only offers old versions, check whether it's a Python-3.14-support gap before
assuming the package or the code is broken.

## Hugging Face downloads hang: `HF_HUB_DISABLE_XET=1`

Downloading `BAAI/bge-m3` via `huggingface_hub`/`FlagEmbedding` hung
indefinitely (0 bytes transferred, no error) using HF's newer "Xet" transfer
backend -- confirmed by watching the `.incomplete` blob file in
`~/.cache/huggingface/hub/` sit at 0 bytes for 15+ minutes. Setting the
environment variable `HF_HUB_DISABLE_XET=1` before the download fixed it,
falling back to plain HTTPS transfer.

**How to apply:** set `HF_HUB_DISABLE_XET=1` in the environment before any
script that pulls a model from Hugging Face Hub on this machine (e.g.
`export HF_HUB_DISABLE_XET=1` or prefix the command). If a download to this
machine ever appears stuck, check the blob's `.incomplete` file size in the
HF cache before assuming it's a network or code issue -- if it's not growing,
this is very likely the cause.

## CUDA PyTorch install for this environment

Neither the CPU-only wheel (`/whl/cpu`, used deliberately in Step 4) nor the
plain PyPI `torch` package has CUDA support for Python 3.14 on this machine
-- the plain PyPI wheel installs as `torch==2.13.0+cpu` in disguise (no
`+cpu` suffix shown, but `torch.cuda.is_available()` is `False`). Checked
`download.pytorch.org`'s CUDA-variant indexes directly: `cu118`, `cu121`,
and `cu124` have **no** `cp314` wheels at all; `cu126`, `cu128`, `cu129`, and
`cu130` do.

**Working install command, confirmed on this machine's GPU (NVIDIA GeForce
RTX 3050 6GB Laptop GPU):**

```
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

This resolves to `torch==2.11.0+cu128` and was verified end-to-end (CUDA
matmul on-device, then a real fp16 model load + inference for
`BAAI/bge-reranker-v2-m3` in Step 7). Note the version is 2.11.0, not the
latest 2.13.x -- cu128 for cp314 hasn't caught up to the newest release yet,
which is fine for our purposes.

**How to apply:** use this exact index/command for any future GPU-dependent
component on this machine -- next up is the LLM in Step 8. Don't assume a
bare `pip install torch` gives you CUDA here; verify with
`torch.cuda.is_available()` after installing.
