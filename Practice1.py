# Colab-ready: dense Farneback optical flow → write & download MP4

# 1) (Re)install headless OpenCV
!pip install --quiet opencv-python-headless

import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files
from IPython.display import clear_output

def draw_flow_hsv(flow):
    """
    Convert flow vectors into an HSV image for visualization.
    Hue = direction, Value = magnitude.
    """
    h, w = flow.shape[:2]
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=True)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[...,0] = (ang / 2).astype(np.uint8)    # Hue: 0..180
    hsv[...,1] = 255                           # Saturation
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def process_video(input_path='input.mp4', output_path='dense_flow.mp4', fps=20):
    """
    Reads input_path (must already exist in Colab workspace),
    computes dense Farneback flow, displays inline,
    writes side-by-side video to output_path, then offers it for download.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    # Read first frame
    ret, prev = cap.read()
    if not ret:
        raise IOError("Cannot read first frame")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Prepare VideoWriter: double width for side-by-side
    h, w = prev.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w*2, h))

    print(f"Processing {input_path} → {output_path} at {fps} FPS...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        flow_bgr = draw_flow_hsv(flow)
        combined = np.hstack([frame, flow_bgr])

        # write and display
        writer.write(combined)
        clear_output(wait=True)
        cv2_imshow(combined)

        prev_gray = gray

    cap.release()
    writer.release()
    print(f"Saved optical flow video to `{output_path}`")
    files.download(output_path)

# Run the processing function.
# Ensure your source MP4 is named 'input.mp4' or change the argument below.
process_video(input_path='input.mp4', output_path='dense_flow.mp4', fps=20)
