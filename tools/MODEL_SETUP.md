# Manual Model Setup for Object Recognition

If the automatic model download fails, follow these steps to manually set up the COCO object detection model.

## Download Model Files

### Option 1: Direct Download

1. **Download MobileNet SSD COCO model (67MB)**
   - URL: http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
   - Save to your computer

2. **Extract the model**
   - Extract the `.tar.gz` file
   - Inside, find `ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb`

3. **Download the config file**
   - URL: https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt
   - Save as `ssd_mobilenet_v2_coco.pbtxt`

## Install Model Files

Place the files in the `tools/models/` directory:

```
ProgramAT/
└── tools/
    ├── models/
    │   ├── frozen_inference_graph.pb       (extracted from tar.gz)
    │   └── ssd_mobilenet_v2_coco.pbtxt     (downloaded config)
    └── object_recognition.py
```

## Verify Installation

Run the test suite:

```bash
cd backend
python test_object_recognition.py
```

You should see output like:
```
COCO model loaded successfully (80 object classes)
Found X objects
Object 1: cell phone (confidence: 0.XX)
Object 2: person (confidence: 0.XX)
```

## What This Enables

With the COCO model installed, the tool can detect **80 object classes** including:

- **Electronics**: cell phone, laptop, mouse, keyboard, remote, tv
- **Common items**: book, bottle, cup, chair, couch, bed
- **Kitchen**: fork, knife, spoon, bowl, microwave, oven, refrigerator
- **And many more...**

Without the model, only basic face detection works (person class only).

## Troubleshooting

**Model doesn't load:**
- Check file paths are correct
- Verify file sizes: `frozen_inference_graph.pb` should be ~67MB
- Ensure OpenCV is installed: `pip install opencv-python`

**Still only detects "person":**
- The model may not be loading - check console output
- Try running with verbose output to see error messages
- Verify both files (`.pb` and `.pbtxt`) are present

**Detection is slow:**
- MobileNet SSD is optimized for CPU but still takes ~200-500ms per frame
- This is normal for real-time object detection on CPU
- Consider throttling frame rate in streaming mode
