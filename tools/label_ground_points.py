"""
===========================================
 Ground Plane Annotator
===========================================

Usage:
------
This tool lets you click on 4 or more points that represent the ground plane.

Features:
- Works with both image and video input.
- Displays instructions on screen.
- Saves clicked points to a JSON file.
- Can be used to annotate multiple frames from a video (e.g., every Nth frame).

Instructions:
-------------
1. Run this script with the path to your image or video.
2. Left-click to mark ground points.
3. Press 'n' to go to the next frame (only for videos).
4. Press 'q' to quit and save all annotations.
5. The coordinates will be saved in 'annotations.json'.

Visually:
---------
  (3)...................(4)
   |                   |
   |                   |
  (1)...................(2)

Example:
--------
python annotate_ground_points.py --input path/to/video.mp4
python annotate_ground_points.py --input frame0.png
"""

import cv2
import json
import os
import argparse
import matplotlib.pyplot as plt
import imghdr


def is_image_file(path):
    """Check if the file is an image based."""
    img_type = imghdr.what(path)
    return img_type is not None

def is_video_file(path):
    """Check if the file is a video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False
    ret, _ = cap.read()
    cap.release()
    return ret

def annotate_frame(frame, frame_idx):
    """Show frame and let user click on points."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points = []

    print(f"\nFrame {frame_idx}: Click on ground points (press ENTER when done)")

    def onclick(event):
        if event.xdata and event.ydata:
            print(f"Clicked: ({event.xdata:.1f}, {event.ydata:.1f})")
            points.append((event.xdata, event.ydata))
            plt.plot(event.xdata, event.ydata, 'ro')
            plt.draw()

    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_title(f"Frame {frame_idx} - Click points, press ENTER when done")
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return points


def main():
    # Load media (image or video)
    input_path = args.input
    annotations = {}

    if is_image_file(input_path):
        # Single image
        frame = cv2.imread(input_path)
        points = annotate_frame(frame, 0)
        annotations["frame_0"] = points

    elif is_video_file(input_path):
        # Video
        cap = cv2.VideoCapture(input_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % args.skip == 0:
                cv2.imshow("Video Preview (press 'a' to annotate, 'q' to quit)", frame)
                key = cv2.waitKey(0) & 0xFF

                if key == ord('a'):
                    points = annotate_frame(frame, frame_idx)
                    annotations[f"frame_{frame_idx}"] = points
                elif key == ord('q'):
                    break
                # any other key skips frame

            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()
    else:
        raise ValueError(f"Unsupported file type or failed to open: {input_path}")

    # Save output
    with open(args.save, "w") as f:
        json.dump(annotations, f, indent=4)
    print(f"\nSaved annotations to {args.save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input image or video.")
    parser.add_argument("--save", default="annotations.json", help="Output JSON file for points.")
    parser.add_argument("--skip", type=int, default=30, help="Process every Nth frame for videos.")
    args = parser.parse_args()

    main()
