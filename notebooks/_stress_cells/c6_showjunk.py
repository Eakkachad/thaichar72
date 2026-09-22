# Look at the junk itself, and at what the model confidently calls it.
#   show_junk("overlap_2_50")   show_junk("touching_2")   show_junk("half_50")
#   show_junk("strokes")        show_junk("blobs")        show_junk("latin_digits")
_rng = np.random.default_rng(0)

def _build_junk(kind, n=200):
    r = np.random.default_rng(0)
    if kind == "touching_2":   return SS.b_pairs(_base_imgs, r, n, 0.0)
    if kind == "overlap_2_25": return SS.b_pairs(_base_imgs, r, n, 0.25)
    if kind == "overlap_2_50": return SS.b_pairs(_base_imgs, r, n, 0.50)
    if kind == "triple":       return SS.b_triples(_base_imgs, r, n)
    if kind == "half_50":      return SS.b_half(_base_imgs, r, n, 0.50)
    if kind == "half_70":      return SS.b_half(_base_imgs, r, n, 0.70)
    if kind == "strokes":      return SS.b_strokes(r, n)
    if kind == "blobs":        return SS.b_blobs(r, n)
    if kind == "latin_digits": return SS.b_latin_digits(r, n)
    if kind == "near_empty":   return SS.b_near_empty(r, n)
    raise ValueError(kind)

def show_junk(kind, n=12, tau=0.9):
    imgs = _build_junk(kind)
    batches, keep = SS.to_batch(imgs, cfg, device)
    pr = SS.probs_for(model, batches)
    pred, conf = pr.argmax(1), pr.max(1)
    print(f"{kind}: {len(keep)} usable of {len(imgs)}   mean confidence {conf.mean():.3f}   "
          f"share at or above {tau}: {(conf >= tau).mean():.1%}")
    order = np.argsort(conf)[::-1][:n]
    cols = min(n, 6); rows = int(np.ceil(len(order) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(1.7 * cols, 2.1 * rows), squeeze=False)
    for axx in np.ravel(ax): axx.axis("off")
    for axx, i in zip(np.ravel(ax), order):
        axx.imshow(imgs[keep[i]], cmap="gray", vmin=0, vmax=255)
        col = "#c0392b" if conf[i] >= tau else "#666666"
        axx.set_title(f"{code_to_char(CLASS_CODES[pred[i]])}" + chr(10) + f"{conf[i]:.2f}",
                      fontsize=10, color=col)
    fig.suptitle(f"{kind} - the model's most confident answers to input with no correct answer",
                 fontsize=11)
    fig.tight_layout(); plt.show()

show_junk("overlap_2_50")
