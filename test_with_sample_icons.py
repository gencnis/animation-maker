"""
Small tester script that runs the morph logic against the two sample
face icons you provided, so you (and we) can verify it works end‑to‑end.

Usage (from project root):

    source .venv/bin/activate  # if using the virtualenv
    python test_with_sample_icons.py
"""

from pathlib import Path

from PIL import Image, ImageOps

from app import create_morph_gif


def main() -> None:
    base = Path("/Users/turkai/.cursor/projects/Users-turkai-Desktop-animationmaker/assets")
    paths = [
        base / "1-5f91e777-eeb5-4bb4-8da5-62615a4d6622.png",
        base / "5-18832b9f-e17c-4112-ade8-e52cb03a276f.png",
    ]

    images = []
    for p in paths:
        print(f"Loading {p}")
        img = Image.open(p)
        img = img.convert("RGBA")
        img = ImageOps.fit(img, (360, 360))
        images.append(img)

    print("Generating morph GIF using shape-based distance-field warping...")
    buf = create_morph_gif(
        images,
        fps=12,
        loop_back=True,
        output_size=360,
        duration_seconds=2.0,
        face_hold_seconds=0.4,
    )
    out_path = Path("sample_morph.gif")
    out_path.write_bytes(buf.getvalue())
    print(f"Done. Wrote {out_path.resolve()} ({out_path.stat().st_size} bytes).")


if __name__ == "__main__":
    main()

