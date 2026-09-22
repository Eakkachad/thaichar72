# Harness sanity check, recomputed live.
#
# The suite's own acceptance criterion is that the clean baseline reproduces the model's known
# validation top-1. Running it the first time is what exposed a real deployment bug: the inference
# path (infer.preprocess_image) scored 2 pt BELOW the training-time dataset transform on the very
# same images, because _reframe padded tight crops with an ink-coloured background and Otsu then
# inverted them. This cell re-measures both paths so the fix stays verified.
from stress_suite import base_sample
from thaichar.infer import load_checkpoint, preprocess_image
from thaichar.data import ThaiGlyphDataset

CKPT = "weights/thaichar72_r18_64_gen.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, cfg = load_checkpoint(CKPT, device=device)
model.eval()

imgs, y = base_sample(8, 0)
keep, xs, gs = [], [], []
for i, a in enumerate(imgs):
    try:
        x, g = preprocess_image(a, cfg)
    except Exception:
        continue
    keep.append(i); xs.append(x); gs.append(g)
keep = np.array(keep); yk = y[keep]
with torch.no_grad():
    p_dep = model(torch.cat(xs).to(device), torch.cat(gs).to(device)).argmax(1).cpu().numpy()

class _Shim:
    def __init__(s, L): s.L = L; s.paths = [str(i) for i in range(len(L))]
    def __len__(s): return len(s.L)
    def __getitem__(s, i): return s.L[i]

sub = [imgs[i] for i in keep]
df = pd.DataFrame({"path": [str(i) for i in range(len(sub))], "label": yk,
                   "height": [a.shape[0] for a in sub], "width": [a.shape[1] for a in sub],
                   "ink_frac": [float((a < 128).mean()) for a in sub]})
ds = ThaiGlyphDataset(df, _Shim(sub), size=cfg["img_size"], channel_mode=cfg["channel_mode"],
                      transform=None, margin=cfg["margin"])
with torch.no_grad():
    p_ds = model(torch.stack([ds[i][0] for i in range(len(ds))]).to(device),
                 torch.stack([ds[i][1] for i in range(len(ds))]).to(device)).argmax(1).cpu().numpy()

print(f"images compared                       : {len(yk)}")
print(f"training-time dataset transform       : {np.mean(p_ds == yk):.4f}")
print(f"deployed preprocess_image (single)    : {np.mean(p_dep == yk):.4f}")
d = p_dep != p_ds
print(f"disagreements                         : {d.sum()}")
print(f"   dataset right / deployed wrong     : {((p_ds == yk) & d).sum()}")
print(f"   deployed right / dataset wrong     : {((p_dep == yk) & d).sum()}")
print()
print("With the _reframe fix these should be within a few images of each other.")
print("Before the fix: 0.9695 vs 0.9898, wrong on 10 and right on none.")
