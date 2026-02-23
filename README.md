# Face Morph GIF Maker

Simple web tool that takes ordered face images, detects facial features, and generates a smooth 360×360 px animated GIF that morphs the shapes of the faces between frames (eyes, mouth, and other features deform into one another). The app runs on your machine and can be reached by anyone on your local network.

## Setup

### 1. Create and activate a virtual environment (recommended)

```bash
cd /Users/turkai/Desktop/animationmaker
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the app

```bash
python app.py
```

The app will start on port `8000` and listen on all interfaces:

- Local machine: `http://localhost:8000/`
- Any device on the same Wi‑Fi/LAN: `http://<your-local-ip>:8000/`

## Finding your local IP on macOS

In a terminal:

```bash
ipconfig getifaddr en0
```

Use the IP that command prints, for example:

- `http://192.168.1.42:8000/`

Anyone on the same network can open that URL in a browser while the app is running.

## Using the tool

1. Open the app in a browser.
2. Upload at least two face images **in the order you want them to morph**.
3. Optionally tweak:
   - **Frames per transition** (smoothness).
   - **Frames per second** (playback speed).
   - Whether to **loop back** smoothly from last face to first.
4. Click **Create GIF**.
5. Wait a moment while the server generates the animation. The 360×360 px GIF will appear on the right side of the page with a download button.

