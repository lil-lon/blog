"""Modal entrypoint for the multi-stage workflow benchmark.

Runs ase / torchsim / alchemi sequentially in one container on the
same A100. Between frameworks we explicitly release GPU memory.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

SUMMARY_JSON = Path("results") / "workflow_summary.json"

app = modal.App("batched-md-workflow")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "build-essential")
    .uv_sync(uv_project_dir="./")
    .add_local_python_source("batched_md")
)


@app.function(image=image, gpu="A100", timeout=7200)
def bench_all() -> dict[str, dict]:
    import gc
    from dataclasses import asdict

    import torch

    from batched_md.workflow_benchmark import RUNNERS

    def _free_gpu() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    plan = ["ase", "torchsim", "alchemi"]
    results: dict[str, dict] = {}
    for framework in plan:
        run = RUNNERS[framework]()
        results[framework] = asdict(run)
        print(f"[{framework}] -> {run}")
        _free_gpu()
    return results


@app.local_entrypoint()
def main() -> None:
    summary = {"workflow": bench_all.remote()}
    out = SUMMARY_JSON.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")
    print(json.dumps(summary, indent=2))
