"""
find_class_mapping_v2.py  --  work out which rock name belongs to each model output index.

What we know: the dataset zip has 53 class folders, the models have 47 outputs.
Training pipelines (e.g. torchvision ImageFolder) number classes 0..N-1 in
ALPHABETICAL folder order, so the 47 trained classes should be the 53 folders with
6 removed, still in alphabetical order. This script:
  1. runs the 5-model ensemble on (all) images of every folder,
  2. shows, for every output index, which folders land on it,
  3. finds the best order-preserving choice of 47 folders (and how certain each choice is),
  4. prints a CLASS_NAMES block with doubtful entries marked.

Usage (from the folder that has infer_ensemble.py and Models/):
    python find_class_mapping_v2.py "D:\\Rover Rock Sample Analysis\\drive-download-20261010T183023Z-1-001.zip"
Optional 2nd argument: max images per folder (default: all).
"""

import io
import sys
import zipfile
from collections import defaultdict

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
NUM_OUT = 47


def best_assignment(P, names, forced_in=(), forced_out=()):
    """P[i][j] = score of folder i (alphabetical order) for output index j.
    Pick exactly len(P[0]) folders, keep alphabetical order, map them to indices
    0..m-1 in increasing order, maximise the summed score. Returns (score, [(i, j)])."""
    n, m = len(P), len(P[0])
    NEG = -1e18
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    take = [[False] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        name = names[i - 1]
        for j in range(m + 1):
            best, tk = NEG, False
            if name not in forced_in and dp[i - 1][j] > NEG / 2:
                best = dp[i - 1][j]
            if j >= 1 and name not in forced_out and dp[i - 1][j - 1] > NEG / 2:
                cand = dp[i - 1][j - 1] + P[i - 1][j - 1]
                if cand > best:
                    best, tk = cand, True
            dp[i][j] = best
            take[i][j] = tk
    if dp[n][m] < NEG / 2:
        return None, None
    pairs, j = [], m
    for i in range(n, 0, -1):
        if take[i][j]:
            pairs.append((i - 1, j - 1))
            j -= 1
    pairs.reverse()
    return dp[n][m], pairs


def report(names, P, counts, totals):
    n, m = len(P), len(P[0])

    print("\n=== A) For every output index: dataset folders that land on it (mean probability) ===")
    for j in range(m):
        col = sorted(((P[i][j], names[i]) for i in range(n)), reverse=True)[:3]
        txt = ", ".join(f"{nm} {p:.2f}" for p, nm in col if p >= 0.05)
        print(f"  {j:2d}: {txt}")

    best, pairs = best_assignment(P, names)
    chosen = {names[i]: j for i, j in pairs}
    excluded = [nm for nm in names if nm not in chosen]

    margins = {}
    for i, nm in enumerate(names):
        if nm in chosen:
            s, _ = best_assignment(P, names, forced_out={nm})
        else:
            s, _ = best_assignment(P, names, forced_in={nm})
        margins[nm] = (best - s) if s is not None else 99.0

    print("\n=== B) Best alphabetical-order assignment ===")
    print("Folders judged NOT used in training (6 expected):")
    for nm in excluded:
        print(f"  {nm:18s} (margin {margins[nm]:.2f})")

    hit = sum(counts[names.index(nm)][j] for nm, j in chosen.items())
    tot = sum(totals[names.index(nm)] for nm in chosen)
    print(f"\nTop-1 hit rate of the 47 mapped folders under this mapping: {100*hit/tot:.1f}%")

    print("\nindex  folder              own-prob  top1-hit  margin   note")
    for nm, j in sorted(chosen.items(), key=lambda x: x[1]):
        i = names.index(nm)
        own = P[i][j]
        t1 = counts[i][j] / totals[i]
        note = []
        if own < 0.2:
            note.append("SUSPICIOUS: models almost never predict this index for this folder")
        if margins[nm] < 0.3:
            note.append("UNCERTAIN")
        print(f"  {j:2d}   {nm:18s}  {own:6.2f}   {t1:6.2f}   {margins[nm]:5.2f}   {'; '.join(note)}")

    print("\n=== C) Paste into infer_ensemble.py ONLY after checking the notes above ===")
    print("CLASS_NAMES = {")
    for nm, j in sorted(chosen.items(), key=lambda x: x[1]):
        i = names.index(nm)
        flag = ""
        if P[i][j] < 0.2:
            flag = "  # SUSPICIOUS"
        elif margins[nm] < 0.3:
            flag = "  # UNCERTAIN"
        print(f'    {j}: "{nm.replace("_", " ")}",{flag}')
    print("}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python find_class_mapping_v2.py path/to/dataset.zip [max_images_per_folder]")
        return
    zip_path = sys.argv[1]
    max_per = int(sys.argv[2]) if len(sys.argv) > 2 else None

    import torch
    from PIL import Image
    from infer_ensemble import DEVICE, MODEL_REGISTRY, load_model, preprocess

    z = zipfile.ZipFile(zip_path)
    by_class = defaultdict(list)
    for nme in sorted(z.namelist()):
        parts = nme.split("/")
        if len(parts) >= 3 and nme.lower().endswith(IMG_EXT):
            by_class[parts[1]].append(nme)
    names = sorted(by_class)
    print(f"Found {len(names)} class folders.")

    print("Loading the 5 models ...")
    models = [load_model(cls, wf) for cls, wf in MODEL_REGISTRY.values()]

    P, counts, totals = [], [], []
    for ci, cname in enumerate(names, start=1):
        files = by_class[cname][:max_per] if max_per else by_class[cname]
        prob_sum = torch.zeros(NUM_OUT)
        cnt = [0] * NUM_OUT
        done, batch = 0, []

        def flush():
            nonlocal prob_sum, done, batch
            if not batch:
                return
            x = torch.stack(batch).to(DEVICE)
            with torch.no_grad():
                probs = torch.stack([torch.softmax(m(x), dim=1) for m in models]).mean(0).cpu()
            prob_sum += probs.sum(0)
            for k in probs.argmax(1).tolist():
                cnt[k] += 1
            done += len(batch)
            batch = []

        for f in files:
            try:
                img = Image.open(io.BytesIO(z.read(f))).convert("RGB")
            except Exception:
                continue
            batch.append(preprocess(img))
            if len(batch) == 32:
                flush()
        flush()

        P.append((prob_sum / max(done, 1)).tolist())
        counts.append(cnt)
        totals.append(max(done, 1))
        top = max(range(NUM_OUT), key=lambda k: cnt[k])
        print(f"  [{ci}/{len(names)}] {cname}: {done} images, most common index {top} ({100*cnt[top]/max(done,1):.0f}%)")

    report(names, P, counts, totals)


if __name__ == "__main__":
    main()
