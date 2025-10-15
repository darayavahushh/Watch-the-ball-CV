import cv2
import json
import numpy as np
import yaml
import os
from ultralytics import YOLO


# Load configuration
CFG = os.getcwd() + '/configs/config.yaml'
with open(CFG, "r") as f:
    cfg = yaml.safe_load(f)

VIDEO_PATH = cfg["inference"]["video"]
WEIGHTS_PATH = cfg["inference"]["weights"]
CONF_THRESH = cfg["inference"]["conf"]
OUTPUT_VIDEO = cfg["inference"]["bev_video"]
ANNOT_JSON = "resources/outputs/coordinates_fixed.json"


def main():
    # Load json
    with open(ANNOT_JSON, 'r') as f:
        annotations = json.load(f)

    # Use the first frame's annotations to define the homography
    first_key = list(annotations.keys())[0]
    img_points = np.array(annotations[first_key][:4], dtype=np.float32)  # first 4 points (corners of ground)

    # Define target BEV plane
    bev_width = 600
    bev_height = 400
    bev_points = np.array([
        [0, bev_height],
        [0, 0],
        [bev_width, 0],
        [bev_width, bev_height]
    ], dtype=np.float32)

    # Compute homography matrix
    H, _ = cv2.findHomography(img_points, bev_points)

    # Initialize YOLO
    model = YOLO(WEIGHTS_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resize original to match BEV height for side-by-side display
    resize_scale = bev_height / height
    new_width = int(width * resize_scale)
    combined_width = new_width + bev_width

    out_writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"XVID"), fps, (combined_width, bev_height))

    # Main functionality
    padding = 100
    canvas_width = bev_width + 2 * padding
    canvas_height = bev_height + 2 * padding
    frame_idx = 0
    tracks = {}  # key: track_id, value: list of (bx, by)
    next_track_id = 0
    prev_centers = []  # centers from previous frame for simple matching
    print(f"Generating side-by-side BEV from: {VIDEO_PATH}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        results = model.predict(frame, conf=CONF_THRESH, verbose=False)
        frame_idx += 1

        # Create blank BEV frame
        bev_frame = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
        bev_frame[:] = (25, 25, 25)
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[padding:padding + bev_height, padding:padding + bev_width] = bev_frame

        # Draw BEV title
        cv2.putText(canvas, "2D BEV Map", (padding + 10, padding - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # Collect current frame centers
        current_centers = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), cls_id, conf in zip(boxes, classes, confs):
                class_name = model.names[cls_id]
                if class_name.lower() not in ["sports ball", "football", "soccer ball"]:
                    continue

                # Compute center of bounding box
                u_center = (x1 + x2) / 2
                v_center = (y1 + y2) / 2

                # Project to BEV
                img_pt = np.array([[[u_center, v_center]]], dtype=np.float32)
                bev_pt = cv2.perspectiveTransform(img_pt, H)[0][0]
                bx, by = int(bev_pt[0]), int(bev_pt[1])
                # flip Y axis to make correct view
                by = bev_height - by
                bx += padding
                by += padding

                current_centers.append((bx, by))

                # Draw detection in BEV
                cv2.circle(canvas, (bx, by), 6, (0, 0, 255), -1)
                cv2.putText(canvas, f"{class_name} {conf:.2f}", (bx + 10, by - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Simple nearest-neighbor tracking
        matched_ids = set()
        for bx, by in current_centers:
            min_dist = float("inf")
            match_id = None
            for track_id, track in tracks.items():
                prev_bx, prev_by = track["trajectory"][-1]
                dist = np.hypot(bx - prev_bx, by - prev_by)
                if dist < 30 and track_id not in matched_ids and dist < min_dist:
                    min_dist = dist
                    match_id = track_id

            if match_id is not None:
                tracks[match_id]["trajectory"].append((bx, by))
                matched_ids.add(match_id)
            else:
                tracks[next_track_id] = {"trajectory": [(bx, by)]}
                next_track_id += 1

        # Draw all trajectories
        for track in tracks.values():
            traj = track["trajectory"]
            if len(traj) > 1:
                for i in range(1, len(traj)):
                    cv2.line(canvas, traj[i - 1], traj[i], (0, 0, 255), 2)


        # Resize the canvas to a fixed width
        canvas_resized = cv2.resize(canvas, (bev_width, bev_height))
        # Resize the original frame to match BEV height
        resize_scale = bev_height / height
        new_width = int(width * resize_scale)
        frame_resized = cv2.resize(frame, (new_width, bev_height))

        # Combine side-by-side
        combined = np.hstack((frame_resized, canvas_resized))

        # Write output
        out_writer.write(combined)

        # Optional live preview
        cv2.imshow("Camera + BEV", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
