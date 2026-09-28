# Security

CleanSplit runs entirely locally: it makes no network calls during inference, and the desktop UI binds to localhost.
The realistic risk surface is therefore small — untrusted **audio files** and untrusted **model checkpoints**.

> **Checkpoints are executable.** `.ckpt` files are pickles. CleanSplit loads its own with `weights_only=True` and
> SHA-256 pins them in `cleansplit/models/checkpoints.py`. Never point it at a checkpoint you did not obtain yourself.

## Reporting

Please use GitHub's **private vulnerability reporting** (Security → Report a vulnerability) rather than a public issue.
A first response should take a few days. This is a personal project with no SLA and no bounty.

## Scope

In scope: code execution from a crafted audio file or config, path traversal in the UI server, anything that makes
CleanSplit reach the network unexpectedly.

Out of scope: vulnerabilities in vendored upstream model code (`third_party/`, `cleansplit/separation/{bs_roformer,mdx23c,scnet}`)
— report those upstream — and the well-known risk of loading an arbitrary pickle you supplied yourself.
