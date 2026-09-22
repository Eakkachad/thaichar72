# Family C - the transform may change the answer, so the output is a MAPPING, not a score.
# Thai is not flip-invariant: mirroring a glyph can produce a different real character.
if len(flip):
    f = flip.copy()
    print("classes that map back onto THEMSELVES under each transform (of 70 present)")
    display(f.pivot_table(index="transform", columns="model", values="maps_to_self", aggfunc="sum"))

    print()
    print("mean confidence under each transform (clean is ~0.9 - a flipped batch shows up here)")
    display(f.pivot_table(index="transform", columns="model", values="mean_conf").round(3))

    silent = f[(~f["maps_to_self"]) & (f["rate"] >= 0.6) & (f["mean_conf"] >= 0.8)]
    g = (silent.groupby(["transform", "true_char", "maps_to_char"]).size()
         .reset_index(name="models_agreeing").sort_values("models_agreeing", ascending=False))
    print()
    print(f"{len(silent)} class x transform combinations become ANOTHER real character confidently.")
    print("These are the silent failures - the model is both wrong and sure.")
    display(g[g.models_agreeing >= 3].reset_index(drop=True))
