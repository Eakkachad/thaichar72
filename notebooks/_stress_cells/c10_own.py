# Run the same checks on YOUR OWN images.
# Point MY_DIR at a folder of glyph images and re-run this cell. Files whose name contains the
# TIS-620 code (e.g. 161ka.png) or a Thai character get scored; the rest are predictions only.
MY_DIR = PROJECT_DIR / "example"

if not Path(MY_DIR).exists():
    print(f"{MY_DIR} does not exist - set MY_DIR to a folder of images and re-run this cell.")
    print("The on-the-day tool is the same code path:")
    print("   uv run --no-sync python scripts/evaluate_folder.py --dir <folder>")
else:
    import subprocess
    cmd = [sys.executable, "scripts/evaluate_folder.py", "--dir", str(MY_DIR),
           "--ckpt", CKPT, "--out-dir", "outputs/onsite_nb"]
    print(" ".join(cmd)); print()
    print(subprocess.run(cmd, capture_output=True, text=True).stdout[-4000:])
