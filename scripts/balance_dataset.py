import os, sys, random, torch

DATA_DIR = "F:/Varroc/data/processed"
INPUT_PT = os.path.join(DATA_DIR, "real_cad_training_data_full.pt")
OUTPUT_PT = os.path.join(DATA_DIR, "real_cad_training_data_balanced.pt")

def main():
    if not os.path.exists(INPUT_PT):
        print(f"Error: {INPUT_PT} not found.")
        sys.exit(1)

    print(f"Loading dataset from {INPUT_PT}...")
    dataset = torch.load(INPUT_PT, map_location='cpu', weights_only=False)
    print(f"Loaded {len(dataset)} graphs.")

    # Filter out invalid graphs with < 3 faces (nodes)
    dataset = [g for g in dataset if g.x.shape[0] >= 3]

    defective = [g for g in dataset if g.y.item() == 1]
    clean     = [g for g in dataset if g.y.item() == 0]

    print(f"Original Count - Defective: {len(defective)}, Clean: {len(clean)}")

    # Undersample defective to 1.2x clean count (55/45 split)
    target_defective = int(len(clean) * 1.2)
    random.seed(42)
    random.shuffle(defective)
    defective_sampled = defective[:target_defective]

    balanced = defective_sampled + clean
    random.shuffle(balanced)

    print(f"Balanced dataset: {len(balanced)} graphs")
    print(f"Defective: {len(defective_sampled)} ({100*len(defective_sampled)/len(balanced):.1f}%)")
    print(f"Clean: {len(clean)} ({100*len(clean)/len(balanced):.1f}%)")

    torch.save(balanced, OUTPUT_PT)
    print(f"Saved balanced dataset to {OUTPUT_PT}")

if __name__ == "__main__":
    main()
