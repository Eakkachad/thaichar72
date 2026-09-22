# Choosing a confidence threshold: what it keeps on real glyphs vs what it rejects on junk.
# Family B showed a threshold is enough to filter invalid input - this is where you pick the number.
_clean_batches, _clean_keep = SS.to_batch(_base_imgs, cfg, device)
_clean_pr = SS.probs_for(model, _clean_batches)
_clean_conf = _clean_pr.max(1)
_clean_ok = _clean_pr.argmax(1) == _base_y[_clean_keep]

JUNK_KINDS = ["overlap_2_50", "touching_2", "half_50", "strokes", "latin_digits"]
_junk_conf = {}
for k in JUNK_KINDS:
    ims = _build_junk(k, 200)
    bt, kp = SS.to_batch(ims, cfg, device)
    _junk_conf[k] = SS.probs_for(model, bt).max(1)

taus = np.linspace(0.3, 0.99, 40)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(_clean_conf, bins=40, alpha=0.75, label="clean glyphs", color="#2b6cb0")
axes[0].hist(np.concatenate(list(_junk_conf.values())), bins=40, alpha=0.75,
             label="invalid input", color="#c0392b")
axes[0].set_yscale("log"); axes[0].set_xlabel("confidence"); axes[0].legend()
axes[0].set_title("where clean and junk sit")

axes[1].plot(taus, [(_clean_conf >= t).mean() for t in taus], label="clean kept", color="#2b6cb0")
for k, v in _junk_conf.items():
    axes[1].plot(taus, [(v >= t).mean() for t in taus], ls="--", lw=1, label=f"{k} leaking")
axes[1].set_xlabel("threshold"); axes[1].set_ylabel("fraction passing")
axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
axes[1].set_title("raise the threshold: clean loss vs junk leaking through")
fig.tight_layout(); plt.show()

print(f"{'threshold':>10} {'clean kept':>12} {'clean acc kept':>16} {'worst junk leaking':>20}")
for t in (0.5, 0.7, 0.8, 0.9, 0.95):
    kept = _clean_conf >= t
    leak = max((v >= t).mean() for v in _junk_conf.values())
    acc = _clean_ok[kept].mean() if kept.any() else float("nan")
    print(f"{t:>10.2f} {kept.mean():>11.1%} {acc:>15.4f} {leak:>19.1%}")
