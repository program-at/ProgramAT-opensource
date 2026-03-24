#!/bin/bash
# Clear all JPEG frames from received_frames directory

FRAMES_DIR="backend/received_frames"

if [ ! -d "$FRAMES_DIR" ]; then
    echo "Directory $FRAMES_DIR does not exist"
    exit 1
fi

# Count frames before deletion
FRAME_COUNT=$(find "$FRAMES_DIR" -name "*.jpg" -o -name "*.jpeg" | wc -l)

if [ $FRAME_COUNT -eq 0 ]; then
    echo "No frames to delete"
    exit 0
fi

echo "Found $FRAME_COUNT frame(s) to delete"

# Delete all JPG/JPEG files
find "$FRAMES_DIR" -name "*.jpg" -delete
find "$FRAMES_DIR" -name "*.jpeg" -delete

echo "Cleared all frames from $FRAMES_DIR"
echo "Disk space freed:"
du -sh "$FRAMES_DIR"
