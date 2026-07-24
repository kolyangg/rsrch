# src/metrics/eval_NS.py
"""
Evaluate ID-similarity and CLIP prompt-similarity for a set of generated images.

Expected layout
---------------
image_folder/            reference images, e.g. lenna.jpg, newton.png …
new_images/              lenna_p0_0.jpg, lenna_p0_1.jpg, newton_p5_2.jpg …
prompt_file              text file: one prompt per line (index = line number)

File-name convention for generated images
-----------------------------------------
{img_basename}_p{p_idx}_{img_id}.jpg
   └──────────┬────────┘ └─┬─┘  └──┬──┘
      person / ref          ↑      any running id (0-n per prompt)
                            │
                   prompt index (0-based, matches line in prompt_file)

Outputs
-------
metrics.csv with columns:
person_id, generated_file, reference_file, prompt_idx, prompt_text,
id_similarity, text_similarity
"""

# very top of eval_NS.py, before "import src..."
import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[2]   # climb 2 levels: src/metrics/…
sys.path.append(str(repo_root))


import argparse, csv, re
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import clip

from src.metrics.text_sim_metric        import TextSimMetric
from src.metrics.id_sim_metric_NS import IDSimOnDemand   # see previous answer

# from text_sim_metric        import TextSimMetric
# from id_sim_metric_NS import IDSimOnDemand   # see previous answer


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
GEN_NAME_RE = re.compile(r"^(?P<base>.+?)_p(?P<pidx>\d+)_(?P<iid>\d+)\.[^.]+$")

def match_reference(ref_dir: Path, base: str) -> Path | None:
    """Return the first image file in ref_dir whose stem == base."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        cand = ref_dir / f"{base}{ext}"
        if cand.exists():
            return cand
    return None

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(image_folder, new_images, prompt_file, out_csv, device):

    # load all prompts
    prompts = [ln.rstrip("\n") for ln in open(prompt_file, encoding="utf-8")]
    print(f"Loaded {len(prompts)} prompts from {prompt_file}")

    id_metric   = IDSimOnDemand(device=device)
    text_metric = TextSimMetric(device=device)

    # pre-encode all prompts for estimated-prompt search
    clip_model, clip_pre = text_metric.model, text_metric.preprocess
    clip_model.eval()
    with torch.no_grad():
        all_tokens    = clip.tokenize(prompts).to(device)
        text_features = clip_model.encode_text(all_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)


    records = []

    gen_paths = sorted(Path(new_images).glob("*"))
    for gen_path in tqdm(gen_paths, total=len(gen_paths),
                         desc="processing images", unit="img"):
    # for gen_path in tqdm(sorted(Path(new_images).glob("*")), desc="generated"):
        m = GEN_NAME_RE.match(gen_path.name)
        if m is None:
            print(f"[WARN]   {gen_path.name} does not follow naming rule – skipped")
            continue

        base   = m["base"]                 # e.g. lenna
        pidx   = int(m["pidx"])            # prompt index
        prompt = prompts[pidx] if pidx < len(prompts) else None
        if prompt is None:
            print(f"[WARN]   prompt idx {pidx} missing for {gen_path.name} – skipped")
            continue

        ref_path = match_reference(Path(image_folder), base)
        if ref_path is None:
            print(f"[WARN]   no reference for {base} – skipped")
            continue

        # debug print ---------------------------------------------------------
        # print(f"[DEBUG]  {gen_path.name:20s} -> ref: {ref_path.name:15s} "
        #       f"prompt[{pidx}]: {prompt[:60]}")

        ref_img = Image.open(ref_path).convert("RGB")
        gen_img = Image.open(gen_path).convert("RGB")

        # id_batch   = {"reference":[ref_img], "generated":[gen_img]}
        id_batch   = {"reference":[ref_img], "generated":[gen_img],
                      "reference_name": ref_path.name,
                      "generated_name": gen_path.name}
        text_batch = {"prompt": prompt,      "generated":[gen_img]}

        with torch.no_grad():
            id_score   = id_metric(**id_batch)
            no_face_flag = id_metric.last_no_face
            text_score = text_metric(**text_batch).item()

            # choose prompt with highest CLIP similarity
            img_tensor = clip_pre(gen_img).unsqueeze(0).to(device)
            img_feat   = clip_model.encode_image(img_tensor)
            img_feat   = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits     = img_feat @ text_features.T
            est_idx    = logits.argmax(dim=-1).item()
            est_prompt = prompts[est_idx]


        records.append(
            [base, gen_path.name, ref_path.name, pidx, prompt, est_prompt,
             id_score, text_score, no_face_flag]
        )

    # save results ------------------------------------------------------------
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            # ["person_id", "generated_file", "reference_file",
            #  "prompt_idx", "real_prompt", "estimated_prompt", "id_similarity", "text_similarity"]
             ["person_id", "generated_file", "reference_file",
             "prompt_idx", "real_prompt", "estimated_prompt",
             "id_similarity", "text_similarity", "no_face_flag"]
        )
        writer.writerows(records)

    print(f"\nWrote {len(records)} rows to {out_csv.resolve()}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated images.")
    parser.add_argument("--image_folder", required=True,
                        help="Folder with source/reference images")
    parser.add_argument("--new_images",   required=True,
                        help="Folder with generated images to score")
    parser.add_argument("--prompt_file",  required=True,
                        help="TXT file with one prompt per line")
    parser.add_argument("--out",          default="results/metrics.csv",
                        help="Output CSV path")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    main(args.image_folder, args.new_images, args.prompt_file,
         args.out, args.device)
