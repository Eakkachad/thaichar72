# Family A - degraded input, the correct answer still exists.
# Retention = top-1 under the corruption divided by that model's own clean top-1.
a = res[res.family == "A"].copy()
a["retention"] = [r.top1 / clean[r.model] for _, r in a.iterrows()]

piv = a.pivot_table(index=["condition", "severity"], columns="model", values="retention")
piv.columns = [c.replace("thaichar72_", "") for c in piv.columns]
print("retention by corruption and severity (1.00 = no loss)")
display(piv.round(3).style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1, axis=None))
