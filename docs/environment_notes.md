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
