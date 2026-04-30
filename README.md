# Coval Benchmarks

Public, open-source voice-AI benchmarking for STT and TTS providers. Runs on a schedule against a pinned dataset, writes results to Postgres, and surfaces them at [benchmarks.coval.ai](https://benchmarks.coval.ai). Licensed Apache-2.0 — clone, build, and run with your own provider API keys.

See [`runner/`](runner/) for setup and local-run instructions.

## Methodology

This is a research benchmark. Results reflect a specific pinned dataset, a
specific set of provider model versions, and a specific normalization pipeline.
Absolute numbers may not generalize to production workloads. Latency metrics
and per-provider drift over time are the most informative signals. See
[`docs/methodology.md`](docs/methodology.md) for the full reproducibility
contract.

The repository is licensed Apache-2.0; the full warranty disclaimer is in
[`LICENSE`](LICENSE) §7. Third-party SDK licenses (e.g. provider client
libraries) are independent of this repository's license — see each SDK's own
license for terms.
