#!/bin/bash
# ----------------------------
# YOLOv8 Video Inference Script
# ----------------------------

# Path to YOLOv8 weights
WEIGHTS="resources/weights/yolov8_person_ball.pt"

# Input video path
VIDEO="resources/rgb.avi"

# Output folder name inside runs/detect
OUTPUT_NAME="inference2D_output_yolo8_person_ball"

# Confidence threshold (0-1)
CONF=0.25

# Run YOLOv8 prediction
yolo task=detect \
     mode=predict \
     model="$WEIGHTS" \
     source="$VIDEO" \
     save=True \
     show=True \
     hide_labels=False \
     conf=$CONF \
     name="$OUTPUT_NAME"

echo "YOLOv8 inference complete!"
echo "Output saved in: runs/detect/$OUTPUT_NAME"