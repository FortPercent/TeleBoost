# Security policy

## Reporting a vulnerability

This codebase is research / training infrastructure: it does not run
adversarially-exposed services, but it does load third-party
checkpoints and execute model code from the wild.

If you find a security issue — for example, a Hydra config-injection
path, a vulnerable dependency we ship, a model-load deserialization
gap, or a credential-leak in commit history — **please do not file a
public issue**.  Email a description and reproduction steps to the
repository owner via the contact information on the GitHub profile of
the owner of this repo, or open a [GitHub Security Advisory] on the
repository.

[GitHub Security Advisory]: https://github.com/FortPercent/TeleBoost/security/advisories

We will:

1. Acknowledge receipt within 5 business days.
2. Investigate and reply with an assessment within 14 business days.
3. Coordinate a fix and disclosure timeline if the issue is valid.

## Supported versions

This is a research codebase, not a maintained product line.  Only
`master` and the most recent named branch (e.g. `import-verl` while
that work is active) receive fixes.  Older tags / branches are
provided as-is.

## Out of scope

* Issues in upstream dependencies (`volcengine/verl`,
  `huggingface/transformers`, `vllm-project/vllm`, `Wan-Video`,
  `tgxs002/HPSv2`, etc.) should be reported to those projects
  directly.  We will pick up upstream fixes as we re-pin the
  `requirements*.txt` files.
* Model weights from third parties (Wan, HPS-v2, DINOv2,
  VideoCLIP-XL, RAFT, VideoPhy) are linked-to but not redistributed
  by this repository.  Their security / provenance is the
  responsibility of the publisher.
