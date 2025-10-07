#!/bin/bash
# ----------------------------
# Labeling Ground Points Script
# ----------------------------

# Input file (image or video)
INPUT="resources/outputs/annotated.avi"

# Output saved file with annotation
OUTPUT="resources/outputs/annotated_ground.json"

# (Optional) Process every n-th frame for videos
FRAMES=1

# Run
python tools/label_ground_points.py --input "$INPUT" --save "$OUTPUT" --skip $FRAMES

echo "Labeled finished!"
echo "Output saved in: $OUTPUT"