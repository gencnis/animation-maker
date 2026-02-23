from io import BytesIO
import base64
from pathlib import Path
from typing import List

import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from PIL import Image, ImageOps


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "change-this-secret-key"

    def _example_images() -> list[str]:
        base_dir = Path(app.root_path) / "images" / "animation-faces"
        if not base_dir.exists():
            return []
        allowed_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
        files: list[str] = []
        for path in sorted(base_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in allowed_suffixes:
                files.append(path.name)
        return files

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html", example_images=_example_images())

    @app.route("/example-images/<path:filename>")
    def example_images(filename: str):
        base_dir = Path(app.root_path) / "images" / "animation-faces"
        return send_from_directory(base_dir, filename)

    @app.route("/generate", methods=["POST"])
    def generate():
        files = request.files.getlist("images")

        try:
            fps = int(request.form.get("fps", 15))
            loop_back = request.form.get("loop_back") == "on"
            output_size = int(request.form.get("resolution", 360))

            duration_seconds_raw = request.form.get("duration_seconds") or ""
            duration_seconds = (
                float(duration_seconds_raw) if duration_seconds_raw.strip() else None
            )

            face_hold_raw = request.form.get("face_hold_seconds") or ""
            face_hold_seconds = (
                float(face_hold_raw) if face_hold_raw.strip() else 0.4
            )
        except ValueError:
            flash(
                "FPS, resolution, and durations must be valid numbers."
            )
            return redirect(url_for("index"))

        # Fixed: fast and smooth transitions (not shown in UI)
        frames_per_transition = 5

        if output_size not in (256, 360, 512, 720):
            output_size = 360

        # Work at a slightly higher internal resolution for smoother edges,
        # then downsample at the very end when writing the GIF.
        work_size = min(output_size * 2, 1024)

        images: List[Image.Image] = []
        for f in files:
            if not f or f.filename == "":
                continue
            try:
                img = Image.open(f.stream)
                img = img.convert("RGBA")
                img = ImageOps.fit(
                    img,
                    (work_size, work_size),
                    Image.LANCZOS,
                    centering=(0.5, 0.5),
                )
                images.append(img)
            except Exception:
                flash(f"Could not read image file: {f.filename}")
                return redirect(url_for("index"))

        # Add example images selected from the gallery (loaded from disk)
        selected_examples = request.form.getlist("example_image")
        if selected_examples:
            base_dir = Path(app.root_path) / "images" / "animation-faces"
            for name in selected_examples:
                path = base_dir / name
                if not path.is_file():
                    continue
                try:
                    img = Image.open(path)
                    img = img.convert("RGBA")
                    img = ImageOps.fit(
                        img,
                        (work_size, work_size),
                        Image.LANCZOS,
                        centering=(0.5, 0.5),
                    )
                    images.append(img)
                except Exception:
                    # Skip unreadable example files and continue with others
                    continue

        if len(images) < 2:
            flash("Please choose at least two images (uploads or examples) to create an animation.")
            return redirect(url_for("index"))

        if len(images) < 2:
            flash("At least two valid image files are required.")
            return redirect(url_for("index"))

        try:
            gif_bytes = create_morph_gif(
                images,
                frames_per_transition=frames_per_transition,
                fps=fps,
                loop_back=loop_back,
                output_size=output_size,
                duration_seconds=duration_seconds,
                face_hold_seconds=face_hold_seconds,
            )
        except ValueError as exc:
            flash(str(exc))
            return redirect(url_for("index"))
        except Exception:
            flash("Unexpected error while generating the animation GIF.")
            return redirect(url_for("index"))

        data_url = base64.b64encode(gif_bytes.getvalue()).decode("ascii")
        return render_template(
            "index.html",
            gif_data=data_url,
            fps=fps,
            loop_back=loop_back,
            resolution=output_size,
            duration_seconds=duration_seconds,
            face_hold_seconds=face_hold_seconds,
            example_images=_example_images(),
        )

    return app


def _shape_morph_pair(
    img1: Image.Image,
    img2: Image.Image,
    frames_per_transition: int,
    include_start: bool = False,
) -> List[Image.Image]:
    """
    Morph two simple white-on-black icon images by interpolating their shapes.

    We convert each image to a binary mask, build a signed distance field for
    each shape, then linearly interpolate those distance fields. Each
    intermediate frame is just the zero-contour of the blended field, which
    causes the white shapes to smoothly deform into one another instead of
    simply fading.
    """
    # Convert to grayscale
    g1 = np.array(img1.convert("L"))
    g2 = np.array(img2.convert("L"))

    # Automatic threshold (Otsu) to separate foreground (white shapes) from background
    _, m1 = cv2.threshold(g1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, m2 = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Normalize to {0,1}
    m1 = (m1 > 0).astype(np.uint8)
    m2 = (m2 > 0).astype(np.uint8)

    # Signed distance fields: positive inside, negative outside
    def signed_distance(mask: np.ndarray) -> np.ndarray:
        inside = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
        return inside - outside

    sd1 = signed_distance(m1)
    sd2 = signed_distance(m2)

    steps = max(1, frames_per_transition)
    frames: List[Image.Image] = []

    # Only interior morph frames; face holds are handled separately.
    for i in range(1, steps):
        alpha = i / float(steps)

        # Blend signed distances and take positive region as foreground
        sd = (1.0 - alpha) * sd1 + alpha * sd2
        mask = (sd > 0).astype(np.uint8) * 255

        # Slightly blur the mask to anti-alias the edges before downsampling
        mask_blurred = cv2.GaussianBlur(mask, (0, 0), sigmaX=0.8, sigmaY=0.8)

        # Create an RGB image: white shapes on black background
        rgb = np.stack([mask_blurred, mask_blurred, mask_blurred], axis=-1).astype(
            np.uint8
        )
        frames.append(Image.fromarray(rgb, mode="RGB").convert("RGBA"))

    return frames


def create_morph_gif(
    images: List[Image.Image],
    frames_per_transition: int = 5,
    fps: int = 15,
    loop_back: bool = True,
    output_size: int = 360,
    duration_seconds: float | None = None,
    face_hold_seconds: float = 0.4,
) -> BytesIO:
    if frames_per_transition < 1:
        frames_per_transition = 1
    if fps < 1:
        fps = 1

    frames: List[Image.Image] = []

    # Approximate how many frames to hold each face based on FPS
    hold_frames = 0
    if face_hold_seconds > 0 and fps > 0:
        hold_frames = max(1, int(round(face_hold_seconds * fps)))

    total = len(images)
    for idx, img in enumerate(images):
        # Hold the current face
        for _ in range(hold_frames):
            frames.append(img.copy())

        # Determine the next face to morph to
        is_last = idx == total - 1
        if is_last and not loop_back:
            continue

        next_img = images[(idx + 1) % total]

        # Add short, smooth transition between current and next face
        if frames_per_transition > 0:
            frames.extend(
                _shape_morph_pair(
                    img,
                    next_img,
                    frames_per_transition=frames_per_transition,
                    include_start=False,
                )
            )

    # Ensure all frames are the requested size and convert to a palette-based format for GIF
    frames = [
        frame.resize((output_size, output_size), Image.LANCZOS).convert(
            "P", palette=Image.ADAPTIVE
        )
        for frame in frames
    ]

    # Duration: either derive from total desired seconds, or from FPS
    if duration_seconds and duration_seconds > 0 and len(frames) > 0:
        duration_ms = max(10, int((duration_seconds * 1000) / len(frames)))
    else:
        duration_ms = int(1000 / fps)
    output = BytesIO()
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    output.seek(0)
    return output


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)

