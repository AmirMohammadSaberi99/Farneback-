# Dense Optical Flow with Farneback in Colab

This notebook-driven script computes dense optical flow between consecutive video frames using OpenCV’s Farneback method. It visualizes flow in HSV color space and writes out a side-by-side comparison video (original vs. flow) as an MP4 file.

## Features
- **Dense Flow Computation**: Uses `cv2.calcOpticalFlowFarneback` for per-pixel motion vectors.
- **HSV Visualization**: Converts flow vectors to an HSV image (hue = direction, value = magnitude) and displays in BGR.
- **Inline Display**: Uses `cv2_imshow` and `clear_output` for live rendering in Colab.
- **Video Output**: Writes the combined original+flow frames to an MP4 file, then provides a download link.

---

## Notebook Setup

1. **Open or create a Colab notebook.** Ensure you have a code cell at the top for dependencies and function definitions.
2. **Place your input video** in the Colab working directory (`/content/`). Rename it to `input.mp4` (or adjust the function argument).

---

## Dependencies

```bash
!pip install --quiet opencv-python-headless
```

- `opencv-python-headless` for image/video processing without X11.
- Colab’s built-in `google.colab.patches.cv2_imshow` for inline display.

---

## Script Breakdown

```python
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files
from IPython.display import clear_output
```

### 1. `draw_flow_hsv(flow)`
- Converts a `(H,W,2)` flow field into an HSV visualization.
- Hue = flow angle (`0–180`), Saturation = 255, Value = normalized magnitude (`0–255`).

### 2. `process_video(input_path, output_path, fps)`

```python
cap = cv2.VideoCapture(input_path)
```
- Opens the input video (must exist in `/content/`).

```python
ret, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
```
- Reads the first frame and converts to grayscale for Farneback.

```python
writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width*2, height)
)
```
- Initializes an MP4 writer with double width (side-by-side frames).

Loop:
```python
for each frame:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, ...)
    flow_bgr = draw_flow_hsv(flow)
    combined = np.hstack([frame, flow_bgr])
    writer.write(combined)
    clear_output(wait=True)
    cv2_imshow(combined)
    prev_gray = gray
```

After loop:
```python
cap.release()
writer.release()
files.download(output_path)
```
- Releases resources and triggers download of the resulting `dense_flow.mp4`.

---

## [Usage Example Video, Click on the link](https://drive.google.com/file/d/1wzoTGsyuu1jL5jDT4uyTsTm9eJyAN48r/view?usp=sharing)

```python
# In a Colab cell:
!pip install --quiet opencv-python-headless
# (Copy the draw_flow_hsv and process_video definitions here)
process_video(
    input_path='input.mp4',
    output_path='dense_flow.mp4',
    fps=20
)
```

- Adjust `input_path` if your file is named differently.
- Modify `fps` to match your source video’s frame rate.

---

## Troubleshooting

- **`Cannot open video file`**: Confirm filename and upload location in `/content/`.
- **Playback lag in Colab**: Lower display resolution or skip inline display to speed up processing.
- **Corrupted output**: Try different `fourcc` codec (e.g. `'XVID'`) or matching frame size exactly.

---

## License

Released under the MIT License. Feel free to adapt or extend!

