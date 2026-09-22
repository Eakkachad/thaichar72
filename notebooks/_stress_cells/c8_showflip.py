# See a flip turn one character into another.
#   show_flip("mirror_v")   show_flip("mirror_h")   show_flip("rot180")   show_flip("rot90_cw")
def show_flip(transform, n=12):
    fn = SS.FAMILY_C[transform]
    imgs = [fn(x) for x in _base_imgs]
    batches, keep = SS.to_batch(imgs, cfg, device)
    pr = SS.probs_for(model, batches)
    pred, conf, yk = pr.argmax(1), pr.max(1), _base_y[keep]
    changed = np.where(pred != yk)[0]
    conf_changed = changed[np.argsort(conf[changed])[::-1]][:n]
    print(f"{transform}: {len(keep)} images, mean confidence {conf.mean():.3f}, "
          f"{(pred == yk).mean():.1%} still read as the original class")
    cols = min(n, 6); rows = int(np.ceil(len(conf_changed) / cols))
    fig, ax = plt.subplots(rows * 2, cols, figsize=(1.7 * cols, 2.2 * rows * 2), squeeze=False)
    for axx in np.ravel(ax): axx.axis("off")
    for j, i in enumerate(conf_changed):
        r, c = divmod(j, cols)
        ax[r * 2][c].imshow(_base_imgs[keep[i]], cmap="gray", vmin=0, vmax=255)
        ax[r * 2][c].set_title(f"original {code_to_char(CLASS_CODES[yk[i]])}", fontsize=9)
        ax[r * 2 + 1][c].imshow(imgs[keep[i]], cmap="gray", vmin=0, vmax=255)
        ax[r * 2 + 1][c].set_title(f"read as {code_to_char(CLASS_CODES[pred[i]])}" + chr(10)
                                   + f"{conf[i]:.2f}", fontsize=9, color="#c0392b")
    fig.suptitle(f"{transform} - top row original, bottom row transformed and what it becomes",
                 fontsize=11)
    fig.tight_layout(); plt.show()

show_flip("mirror_v")
