#!/usr/bin/env python3
"""Build a compact PM0/CL19/CL23 hard-case face comparison from sealed sheets."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "analysis/assets/cl21_cl26_20260814_current"
RUNS = ("PM0", "CL19", "CL23")
PROMPTS = ("Skiing", "Crying")
IDENTITIES = ("Eddie", "Elon", "Jennie", "Jensen", "Jisoo", "Keanu", "Lex", "Marion")
SOURCE_SIDE = 90
SOURCE_HEADER = 42
SOURCE_TILE = 340
TARGET_SIDE = 112
TARGET_HEADER = 58
TARGET_TILE = 148


def font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    sheets = {
        run: Image.open(ASSETS / f"hardcases_{run}_face_crops.jpg").convert("RGB")
        for run in RUNS
    }
    canvas = Image.new(
        "RGB",
        (TARGET_SIDE + len(RUNS) * len(PROMPTS) * TARGET_TILE,
         TARGET_HEADER + len(IDENTITIES) * TARGET_TILE),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for run_index, run in enumerate(RUNS):
        for prompt_index, prompt in enumerate(PROMPTS):
            column = run_index * len(PROMPTS) + prompt_index
            x = TARGET_SIDE + column * TARGET_TILE
            draw.text((x + 5, 5), run, fill="black", font=font(18))
            draw.text((x + 5, 29), prompt, fill="black", font=font(14))
    for row, identity in enumerate(IDENTITIES):
        y = TARGET_HEADER + row * TARGET_TILE
        draw.text((6, y + 10), identity, fill="black", font=font(16))
        for run_index, run in enumerate(RUNS):
            sheet = sheets[run]
            for prompt_index, _ in enumerate(PROMPTS):
                left = SOURCE_SIDE + prompt_index * SOURCE_TILE
                top = SOURCE_HEADER + row * SOURCE_TILE
                tile = sheet.crop((left, top, left + SOURCE_TILE, top + SOURCE_TILE))
                tile = tile.resize((TARGET_TILE, TARGET_TILE), Image.Resampling.LANCZOS)
                column = run_index * len(PROMPTS) + prompt_index
                canvas.paste(tile, (TARGET_SIDE + column * TARGET_TILE, y))
    output = ASSETS / "hardcases_pm0_cl19_cl23_face_comparison.jpg"
    canvas.save(output, quality=94, optimize=True)
    print(output)


if __name__ == "__main__":
    main()
