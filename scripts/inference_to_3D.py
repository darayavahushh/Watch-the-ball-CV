# inference_3D.py
import yaml
import cv2
import csv
import os
import numpy as np
from ultralytics import YOLO
from tools.pixel_coords import CameraCoords
from tools.object_geometry import Ball3DReconstruct
from tools.object_tracker import ObjectTracker


# Load configuration
CFG = '/home/kyouma/Codes/Watch-the-ball-CV/configs/config.yaml'
with open(CFG, "r") as f:
    cfg = yaml.safe_load(f)

VIDEO_PATH = cfg["inference"]["video"]
WEIGHTS_PATH = cfg["inference"]["weights"]
CSV_OUT = cfg["inference"]["csv_out"]
OUT_VIDEO = cfg["inference"]["out_video"]
CONF_THRESH = cfg["inference"]["conf"]
DEVICE = cfg["inference"].get("device", "cuda")
DRAW = cfg["inference"].get("draw", True)

# Camera intrinsics
FX, FY = cfg["inference"]["fx"], cfg["inference"]["fy"]
CX, CY = cfg["inference"]["cx"], cfg["inference"]["cy"]

# Football diameter in meters (standard size)
BALL_RADIUS = cfg["objects"]["football_radius"]  # meters
BALL_DIAMETER = 2 * BALL_RADIUS

# Global tracking state
trajectory_points = []

def draw_3d_detection(frame, bbox, class_name, distance, X, Y, Z, fx, fy, cx, cy, trajectory=None, color=(0, 255, 0)):
    """
    Draws an oriented 3D bounding box and trajectory for a detected ball.

    Args:
        frame: np.ndarray
        bbox: [x1, y1, x2, y2]
        class_name: str
        distance: float (m)
        X, Y, Z: float (m)
        fx, fy, cx, cy: camera intrinsics
        color: tuple (B, G, R)
    """
    global trajectory_points #, prev_pos

    # # Compute orientation (motion direction)
    # if prev_pos is not None:
    #     velocity = np.array([X, Y, Z], dtype=float) - np.array(prev_pos, dtype=float)
    #     v_norm = np.linalg.norm(velocity)
    #     if v_norm > 1e-5:
    #         yaw = np.arctan2(velocity[0], velocity[2])  # rotation around Y-axis
    #     else:
    #         yaw = smooth_yaw  # keep last rotation if almost still
    #
    #     # Smooth yaw updates
    #     alpha = 0.15 if v_norm > 0.02 else 0.05  # slower smoothing if still
    #     smooth_yaw = (1 - alpha) * smooth_yaw + alpha * yaw
    # else:
    #     smooth_yaw = 0.0
    #     v_norm = 0.0
    #
    # prev_pos = np.array([X, Y, Z], dtype=float)

    # Define cube corners in local coordinates ===
    r = BALL_RADIUS
    corners_3d = np.array([
        [-r, -r, -r],   # bottom-back-left
        [ r, -r, -r],   # bottom-back-right
        [ r,  r, -r],   # top-back-right
        [-r,  r, -r],   # top-back-left
        [-r, -r,  r],   # bottom-front-left
        [ r, -r,  r],   # bottom-front-right
        [ r,  r,  r],   # top-front-right
        [-r,  r,  r]    # top-front-left
    ])

    # Apply camera-relative transform (no rotation, too hard for a football with no "front face")
    pitch = np.deg2rad(min(10, distance * 2))  # tilt up to 10° max
    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    corners_3d = (rot_x @ corners_3d.T).T + np.array([X, Y, Z])
    # corners_3d += np.array([X, Y, Z])

    # Project 3D points to 2D ===
    projected = []
    for (Xp, Yp, Zp) in corners_3d:
        if Zp <= 0:
            continue
        u = int((Xp * fx / Zp) + cx)
        v = int((Yp * fy / Zp) + cy)
        projected.append((u, v))

    if len(projected) != 8:
        return  # skip if invalid projection

    # Define cube edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7)   # verticals
    ]

    # Draw edges
    for (i, j) in edges:
        cv2.line(frame, projected[i], projected[j], color, 2)

    # Cube center in image
    u_center = int((projected[0][0] + projected[6][0]) / 2)
    v_center = int((projected[0][1] + projected[6][1]) / 2)

    # Track trajectory of center point
    if trajectory is not None and len(trajectory) > 1:
        for i in range(1, len(trajectory)):
            cv2.line(frame,
                     (int(trajectory[i - 1][0]), int(trajectory[i - 1][1])),
                     (int(trajectory[i][0]), int(trajectory[i][1])),
                     (0, 255, 255), 2)

    # Display label
    label = f"{class_name} {distance:.2f}m"
    top_y = min([p[1] for p in projected])  # smallest v coordinate (top of the cube)
    top_x = np.mean([p[0] for p in projected])  # center horizontally
    label_y_offset = 15  # pixels offset above
    cv2.putText(frame, label, (int(top_x) - 60, int(top_y) - label_y_offset),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, lineType=cv2.LINE_AA)

    # Update previous position
    # prev_pos = (X, Y, Z)


def main():
    # Initialize YOLO model
    model = YOLO(WEIGHTS_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize geometry classes
    coords = CameraCoords(width, height, FX, FY, CX, CY)
    reconstructor = Ball3DReconstruct(coords, BALL_RADIUS)

    # Initialize tracker
    tracker = ObjectTracker(max_lost=10, distance_thresh=80)

    # Prepare output directories
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_VIDEO), exist_ok=True)

    # CSV and video output
    csv_file = open(CSV_OUT, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_idx", "time_s", "class_name", "conf",
        "x1", "y1", "x2", "y2",
        "X_m", "Y_m", "Z_m", "distance_m"
    ])

    out_writer = None
    if DRAW:
        out_writer = cv2.VideoWriter(
            OUT_VIDEO,
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (width, height)
        )

    print(f"Running 3D inference on: {VIDEO_PATH}")
    frame_idx = 0
    compensation_pos = {}  # track_id -> (X, Y, Z)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        results = model.predict(frame, conf=CONF_THRESH, device=DEVICE, verbose=False)
        time_s = frame_idx / fps

        detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, classes):
                class_name = model.names[cls_id]
                if class_name.lower() not in ["sports ball", "football", "soccer ball"]:
                    continue

                bbox = [x1, y1, x2, y2]
                X, Y, Z = reconstructor.compute_3d_position(bbox)
                distance = reconstructor.compute_distance(X, Y, Z)

                # store center and bbox
                u = (x1 + x2) / 2
                v = (y1 + y2) / 2
                detections.append((u, v, bbox, class_name, conf, X, Y, Z, distance))

                # Write to CSV
                csv_writer.writerow([
                    frame_idx, round(time_s, 3), class_name, round(float(conf), 3),
                    round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
                    round(X, 3), round(Y, 3), round(Z, 3), round(distance, 3)
                ])

        # Camera motion compensation (sadly doesn't work when there is only 1 detection)
        # More complex RANSAC variant produces poor results with only 1 detection.
        if frame_idx > 0 and len(detections) > 0:
            # Get previous positions for matched IDs
            prev_xyz = []
            curr_xyz = []

            # Match by closest 2D center
            for track_id, track in tracker.tracks.items():
                if "last_xyz" in track:
                    prev_xyz.append(track["last_xyz"])
            curr_xyz = np.array([[d[5], d[6], d[7]] for d in detections])  # current frame XYZ

            if len(prev_xyz) > 0:
                prev_xyz = np.array(prev_xyz)
                # Compute median motion vector
                median_motion = np.median(curr_xyz - prev_xyz[:len(curr_xyz)], axis=0)
            else:
                median_motion = np.zeros(3)

            # Apply compensation to all current detections
            compensated_detections = []
            for d in detections:
                u, v, bbox, class_name, conf, X, Y, Z, distance = d
                X_c, Y_c, Z_c = X - median_motion[0], Y - median_motion[1], Z - median_motion[2]
                compensated_detections.append((u, v, bbox, class_name, conf, X_c, Y_c, Z_c, distance))
        else:
            compensated_detections = detections

        # Update tracker & draw each track
        tracked = tracker.update([(d[0], d[1], d[2]) for d in compensated_detections])

        for track_id, u, v, bbox in tracked:
            # Find corresponding detection
            for d in compensated_detections:
                if abs(d[0] - u) < 2 and abs(d[1] - v) < 2:
                    _, _, bbox, class_name, conf, X, Y, Z, distance = d
                    # Draw with ID and trajectory
                    draw_3d_detection(
                        frame, bbox, f"{class_name} ID:{track_id}",
                        distance, X, Y, Z, FX, FY, CX, CY,
                        trajectory=tracker.tracks[track_id]["trajectory"]
                    )
                    break

        if DRAW and out_writer:
            out_writer.write(frame)

        frame_idx += 1

    # Cleanup
    cap.release()
    csv_file.close()
    if out_writer:
        out_writer.release()

    print("\nInference complete.")
    print(f"CSV results saved to: {CSV_OUT}")
    if DRAW:
        print(f"Annotated video saved to: {OUT_VIDEO}")


if __name__ == "__main__":
    main()