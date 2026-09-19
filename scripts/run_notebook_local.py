#!/usr/bin/env python3
"""Acceptance harness: execute notebooks/ThaiChar72_Colab.ipynb headlessly.

Usage:
    uv run python scripts/run_notebook_local.py \
        --mode inference \
        --weights runs/B_full_resnet18_64_T4/best.pt

The script:
1. Regenerates the notebook from scripts/build_notebook.py.
2. Patches the config cell to inject MODE, USE_DRIVE, WEIGHTS_PATH, PROJECT_DIR.
3. Executes every cell via nbclient (timeout 900 s/cell) using the 'thaichar' kernel.
4. Saves the executed notebook to notebooks/ThaiChar72_Colab.executed.ipynb.
5. Prints a summary and exits non-zero if any error output is found.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running without uv prefix by ensuring the project venv is used
# ---------------------------------------------------------------------------
import importlib.util

_REQUIRED = ["nbclient", "nbformat"]
for _pkg in _REQUIRED:
    if importlib.util.find_spec(_pkg) is None:
        print(f"[run_notebook_local] Installing missing package: {_pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pkg])

import nbclient  # noqa: E402
import nbformat  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Execute ThaiChar72 notebook locally.")
    p.add_argument(
        "--mode",
        default="inference",
        choices=["inference", "train"],
        help="Execution mode (default: inference)",
    )
    p.add_argument(
        "--weights",
        default=None,
        help="Path to best.pt; defaults to runs/B_full_resnet18_64_T4/best.pt "
        "then runs/A1_resnet18_full_64_T4/best.pt",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-cell timeout in seconds (default: 900)",
    )
    p.add_argument(
        "--kernel",
        default="thaichar",
        help="Jupyter kernel name (default: thaichar)",
    )
    p.add_argument(
        "--nb-in",
        default="notebooks/ThaiChar72_Colab.ipynb",
        help="Input notebook path (default: notebooks/ThaiChar72_Colab.ipynb)",
    )
    p.add_argument(
        "--nb-out",
        default="notebooks/ThaiChar72_Colab.executed.ipynb",
        help="Output executed notebook path",
    )
    p.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Skip rebuilding the notebook before execution",
    )
    return p.parse_args()


def resolve_weights(root: Path, weights_arg: str | None) -> Path:
    if weights_arg:
        w = Path(weights_arg)
        if not w.is_absolute():
            w = root / w
        if w.exists():
            return w
        print(f"[WARNING] --weights {w} not found; falling back to defaults.")

    candidates = [
        root / "runs" / "B_full_resnet18_64_T4" / "best.pt",
        root / "runs" / "A1_resnet18_full_64_T4" / "best.pt",
        root / "weights" / "best.pt",
        root / "best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No checkpoint found. Tried: {candidates}"
    )


def patch_config_cell(nb: nbformat.NotebookNode, mode: str, weights: Path, root: Path) -> None:
    """Inject config overrides at the TOP of the config cell (cell index 1 among code cells).

    We prepend an override block so it runs before the existing defaults, then
    the existing code re-reads them (where it uses globals().get / checks exists).
    """
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 2, "Expected at least 2 code cells"
    config_cell = code_cells[1]

    override_block = (
        f"# ---- INJECTED BY run_notebook_local.py ----\n"
        f"MODE = {mode!r}\n"
        f"USE_DRIVE = False\n"
        f"WEIGHTS_PATH = __import__('pathlib').Path({str(weights)!r})\n"
        f"PROJECT_DIR = __import__('pathlib').Path({str(root)!r})\n"
        f"# ---- END INJECTION ----\n\n"
    )

    existing = config_cell.source if isinstance(config_cell.source, str) else "".join(config_cell.source)
    config_cell.source = override_block + existing

    print(f"[patch_config_cell] Injected MODE={mode!r}, WEIGHTS_PATH={weights}, PROJECT_DIR={root}")


def ensure_kernel(kernel_name: str, venv_python: str) -> None:
    """Register the ipykernel if not already installed."""
    import subprocess as sp

    result = sp.run(
        ["jupyter", "kernelspec", "list", "--json"],
        capture_output=True, text=True,
    )
    try:
        specs = json.loads(result.stdout).get("kernelspecs", {})
    except Exception:
        specs = {}

    if kernel_name not in specs:
        print(f"[ensure_kernel] Kernel '{kernel_name}' not found; installing...")
        sp.check_call(
            [venv_python, "-m", "ipykernel", "install", "--user",
             "--name", kernel_name, "--display-name", f"ThaiChar72 (Python 3.12)"],
        )
        print(f"[ensure_kernel] ✓ Kernel '{kernel_name}' installed.")
    else:
        print(f"[ensure_kernel] ✓ Kernel '{kernel_name}' already registered.")


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent  # project root

    # ------------------------------------------------------------------
    # Step 1: Rebuild notebook
    # ------------------------------------------------------------------
    if not args.no_rebuild:
        print("[run_notebook_local] Rebuilding notebook from scripts/build_notebook.py ...")
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / "build_notebook.py")],
            cwd=str(root),
        )
        if result.returncode != 0:
            print("[ERROR] build_notebook.py failed.")
            return 1
        print("[run_notebook_local] ✓ Notebook rebuilt.")

    # ------------------------------------------------------------------
    # Step 2: Resolve weights
    # ------------------------------------------------------------------
    weights = resolve_weights(root, args.weights)
    print(f"[run_notebook_local] Using weights: {weights}")

    # ------------------------------------------------------------------
    # Step 3: Load and patch notebook
    # ------------------------------------------------------------------
    nb_in = root / args.nb_in
    nb_out = root / args.nb_out

    print(f"[run_notebook_local] Loading notebook: {nb_in}")
    with open(nb_in, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    patch_config_cell(nb, mode=args.mode, weights=weights, root=root)

    # ------------------------------------------------------------------
    # Step 4: Ensure kernel
    # ------------------------------------------------------------------
    ensure_kernel(args.kernel, sys.executable)

    # ------------------------------------------------------------------
    # Step 5: Execute
    # ------------------------------------------------------------------
    print(f"[run_notebook_local] Executing notebook (kernel={args.kernel}, timeout={args.timeout}s) ...")
    t0 = time.perf_counter()

    ep = nbclient.NotebookClient(
        nb,
        timeout=args.timeout,
        kernel_name=args.kernel,
        resources={"metadata": {"path": str(root)}},
        allow_errors=False,  # surface errors; we check afterwards
    )

    try:
        ep.execute()
        elapsed = time.perf_counter() - t0
        print(f"[run_notebook_local] ✓ Execution completed in {elapsed:.1f}s")
        exec_ok = True
    except nbclient.exceptions.CellExecutionError as exc:
        elapsed = time.perf_counter() - t0
        print(f"[run_notebook_local] ✗ CellExecutionError after {elapsed:.1f}s:\n{exc}")
        exec_ok = False

    # ------------------------------------------------------------------
    # Step 6: Save executed notebook (always, even on failure)
    # ------------------------------------------------------------------
    nb_out.parent.mkdir(parents=True, exist_ok=True)
    with open(nb_out, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"[run_notebook_local] Saved executed notebook → {nb_out}")

    # ------------------------------------------------------------------
    # Step 7: Count error outputs
    # ------------------------------------------------------------------
    with open(nb_out, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    error_outputs = [
        o
        for c in nb_data["cells"]
        if c["cell_type"] == "code"
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    ]

    print(f"\n[run_notebook_local] Error outputs in executed notebook: {len(error_outputs)}")
    for eo in error_outputs:
        print(f"  • {eo.get('ename')}: {eo.get('evalue')}")

    if not exec_ok or error_outputs:
        print("[run_notebook_local] ✗ Notebook execution had errors. See above.")
        return 1

    print("[run_notebook_local] ✓ All cells executed successfully with zero error outputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
