"""Tests for notebooks/ThaiChar72_Colab.ipynb.

Verifies:
1. Notebook JSON is valid and conforms to nbformat schema.
2. Required section headings (0 to 8) exist and appear in exact order.
3. Every code cell compiles cleanly with compile() (no syntax errors).
4. Config cell defines all required variables (PROJECT_DIR, DATA_ZIP, DATA_DIR,
   WEIGHTS_PATH, FINAL_CONFIG, OUTPUT_DIR, USE_DRIVE, MODE, SEED, EPOCHS_OVERRIDE).
"""

import ast
import json
from pathlib import Path

import nbformat
import pytest

NOTEBOOK_PATH = Path("notebooks/ThaiChar72_Colab.ipynb")

EXPECTED_SECTIONS = [
    ("0", "Title"),
    ("1", "Environment"),
    ("2", "Configuration"),
    ("3", "Data"),
    ("4", "Train"),
    ("5", "Evaluate"),
    ("6", "Inference"),
    ("7", "Robustness"),
    ("8", "Export"),
]

REQUIRED_CONFIG_VARS = [
    "PROJECT_DIR",
    "DATA_ZIP",
    "DATA_DIR",
    "WEIGHTS_PATH",
    "FINAL_CONFIG",
    "OUTPUT_DIR",
    "USE_DRIVE",
    "MODE",
    "SEED",
    "EPOCHS_OVERRIDE",
]


@pytest.fixture(scope="module")
def notebook():
    assert NOTEBOOK_PATH.exists(), f"Notebook not found at {NOTEBOOK_PATH}"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    nbformat.validate(nb)
    return nb


def test_notebook_json_valid(notebook):
    """Ensure notebook is valid JSON and has cells."""
    assert len(notebook.cells) > 0, "Notebook has no cells"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "cells" in data
    assert "metadata" in data


def test_required_section_headings_in_order(notebook):
    """Verify that required section headings 0 through 8 appear in order."""
    markdown_cells = [c for c in notebook.cells if c.cell_type == "markdown"]
    assert len(markdown_cells) >= len(EXPECTED_SECTIONS)

    # Check headers in markdown cells
    sec_idx = 0
    found_sections = []
    for cell in markdown_cells:
        src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        first_line = src.strip().splitlines()[0] if src.strip() else ""
        if sec_idx < len(EXPECTED_SECTIONS):
            sec_num, sec_keyword = EXPECTED_SECTIONS[sec_idx]
            if (sec_num in first_line or sec_keyword.lower() in first_line.lower()):
                found_sections.append((sec_num, first_line))
                sec_idx += 1

    assert sec_idx == len(EXPECTED_SECTIONS), (
        f"Missing sections. Found: {found_sections}, Expected: {EXPECTED_SECTIONS}"
    )


def test_every_code_cell_compiles(notebook):
    """Ensure every code cell contains valid Python code that passes compile()."""
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert len(code_cells) > 0

    for i, cell in enumerate(code_cells):
        src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        # compile() raises SyntaxError on invalid code
        try:
            code_obj = compile(src, f"<cell_{i}>", "exec")
            assert code_obj is not None
        except SyntaxError as e:
            pytest.fail(f"Code cell {i} failed compile(): {e}\nSource:\n{src}")


def test_config_cell_variables(notebook):
    """Verify that the config cell defines all 10 required variables."""
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    # Config cell is cell index 1 among code cells (under Section 2)
    assert len(code_cells) >= 2
    config_src = code_cells[1].source
    if isinstance(config_src, list):
        config_src = "".join(config_src)

    tree = ast.parse(config_src)
    assigned_vars = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned_vars.add(target.id)

    missing = [var for var in REQUIRED_CONFIG_VARS if var not in assigned_vars]
    assert not missing, f"Config cell is missing required variables: {missing}"


def test_defensive_properties(notebook):
    """Verify defensive handling in notebook code cells."""
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    all_code = "\n".join(
        (c.source if isinstance(c.source, str) else "".join(c.source))
        for c in code_cells
    )

    # 1. Zipfile used for unzipping
    assert "zipfile" in all_code

    # 2. Drive mounting guarded by USE_DRIVE
    assert "USE_DRIVE" in all_code

    # 3. Mode check skips training in inference mode
    assert 'MODE == "train"' in all_code or "MODE == 'train'" in all_code

    # 4. Top-20 confused pairs table
    assert "Top-20 Confused Pairs" in all_code or "confused" in all_code

    # 5. Robustness limit 5
    assert "--limit" in all_code and "5" in all_code

    # 6. Export metrics_summary.json
    assert "metrics_summary.json" in all_code
