#!/bin/bash
# ----------------------------
# Labeling Ground Points Script
# ----------------------------

# Input file (image or video)
INPUT="resources/outputs/annotated.avi"
# In this videos case there will be 5 points to label a frame
#      (5).....................(4)
#     |                         |
#  (1)                          |
#     |                         |
#     (2).....................(3)

# Output saved file with annotation
OUTPUT="resources/outputs/annotated_ground.json"

# (Optional) Process every n-th frame for videos
FRAMES=3

# Run
python tools/label_ground_points.py --input "$INPUT" --save "$OUTPUT" --skip $FRAMES

echo "Labeled finished!"
echo "Output saved in: $OUTPUT"