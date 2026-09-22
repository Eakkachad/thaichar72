# Family B - invalid input. There is NO correct answer, so nothing here reports accuracy.
# separability AUROC = can a confidence threshold alone tell this junk from clean input?
#   >= 0.8 yes, a threshold works.   <= 0.6 no, the input pipeline has to be fixed instead.
b = res[res.family == "B"].copy()
b["model"] = b.model.str.replace("thaichar72_", "", regex=False)

print("separability AUROC (higher = easier to filter out by confidence)")
display(b.pivot_table(index="condition", columns="model", values="separability_auroc")
         .round(3).style.background_gradient(cmap="RdYlGn", vmin=0.5, vmax=1.0, axis=None))

print()
print("fraction of junk that still gets a >= 0.9 confidence prediction (lower = safer)")
display(b.pivot_table(index="condition", columns="model", values="confident_wrong@0.9")
         .round(3).style.background_gradient(cmap="RdYlGn_r", vmin=0, vmax=0.25, axis=None))

worst = b.loc[b.separability_auroc.idxmin()]
print()
print(f"weakest separation: {worst.condition} at AUROC {worst.separability_auroc:.3f}")
print("A splitter for touching glyphs is only worth building if some family sits at or below 0.6.")
