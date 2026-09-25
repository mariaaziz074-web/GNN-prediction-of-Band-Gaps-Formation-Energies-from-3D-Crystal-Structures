from jarvis.db.figshare import data

print("[INFO] Loading JARVIS-DFT 3D dataset...")
records = data("dft_3d")

print(f"[OK] Records loaded: {len(records):,}")

if not records:
    raise RuntimeError("JARVIS returned no records.")

sample = records[0]

print("\nAvailable fields:")
for key in sorted(sample.keys()):
    value = sample[key]
    if key == "atoms":
        print(f"  {key}: <structure dictionary>")
    else:
        text = str(value)
        print(f"  {key}: {text[:120]}")

print("\nCandidate target fields:")
for key in sample:
    k = key.lower()
    if any(term in k for term in ["gap", "formation", "energy"]):
        print(f"  {key}: {sample[key]}")
