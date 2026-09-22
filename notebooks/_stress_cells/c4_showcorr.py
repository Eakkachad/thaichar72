# See what a corruption actually does, and what the model says about it.
#   show_corruption("pixelate", 0.25)      show_corruption("rotate_hard", 30)
#   show_corruption("stroke_extreme", -4)  show_corruption("unseen_font")
import stress_suite as SS
from thaichar.classes import CLASS_CODES, code_to_char

_base_imgs, _base_y = base_sample(8, 0)

def show_corruption(name, severity=None, n=12, ckpt=None):
    if ckpt is None:
        m, c = model, cfg
    else:
        m, c = load_checkpoint(ckpt, device=device); m.eval()
    if name == "unseen_font":
        imgs, ys, nf = SS.render_unseen_font(1, 0)
        title = f"unseen_font ({nf} typefaces the stage-1 render never used)"
    else:
        fn, sevs = SS.FAMILY_A[name]
        if severity is None:
            severity = sevs[-1]
        imgs = [fn(x, severity) for x in _base_imgs]
        ys = _base_y
        title = f"{name} severity={severity}"
    batches, keep = SS.to_batch(imgs, c, device)
    if len(keep) == 0:
        print("every image was rejected by preprocessing"); return
    pr = SS.probs_for(m, batches)
    pred, conf, yk = pr.argmax(1), pr.max(1), np.array(ys)[keep]
    acc = float((pred == yk).mean())
    print(f"{title}   top-1 {acc:.4f}   mean confidence {conf.mean():.3f}   "
          f"rejected by preprocessing {len(imgs) - len(keep)}")
    order = np.argsort(conf)[::-1]
    wrong = [i for i in order if pred[i] != yk[i]][:n]
    if not wrong:
        print("no errors at this severity"); return
    cols = min(n, 6); rows = int(np.ceil(len(wrong) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(1.7 * cols, 2.1 * rows), squeeze=False)
    for axx in np.ravel(ax): axx.axis("off")
    for axx, i in zip(np.ravel(ax), wrong):
        axx.imshow(imgs[keep[i]], cmap="gray", vmin=0, vmax=255)
        axx.set_title(f"{code_to_char(CLASS_CODES[yk[i]])} to "
                      f"{code_to_char(CLASS_CODES[pred[i]])}" + chr(10) + f"{conf[i]:.2f}",
                      fontsize=9, color="#c0392b")
    fig.suptitle("most-confident mistakes under " + title, fontsize=11)
    fig.tight_layout(); plt.show()

show_corruption("pixelate", 0.25)
