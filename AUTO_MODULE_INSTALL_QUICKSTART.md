# Automatic Module Installation - Quick Start

## ✅ What's New

Your backend now **automatically installs missing Python modules** when tools need them. No more manual backend code changes!

## 🚀 How to Use

### For Tool Developers

Just write your tool code normally and import whatever you need:

```python
# tools/my_advanced_tool.py
import scipy
from skimage import filters
import pandas as pd

def main(image, input_data):
    # Your code here
    return "Processing complete"
```

**That's it!** The first time this tool runs:
1. Missing modules are detected automatically
2. They're installed via pip
3. The tool runs successfully
4. Future runs are instant (modules are cached)

### For Users

Nothing changes! Just run tools as normal:
1. Select your issue
2. Select your tool
3. Tap "▶️ Process Image"

If the tool needs new modules, you might see a 2-5 second delay on first run while they install. After that, it's instant.

## 📦 Pre-Installed Modules

These are **always available** (no installation delay):

- `cv2` (OpenCV)
- `np` (NumPy)
- `Image` (PIL/Pillow)
- `pytesseract` (OCR)
- `datetime`
- `json`
- `math`
- `re`
- `scipy` (if installed)
- `sklearn` (scikit-learn, if installed)

## 🔧 How It Works

```
Tool imports module
       ↓
Module not found?
       ↓
pip install <module>
       ↓
Add to requirements.txt
       ↓
Retry tool execution
       ↓
Success! ✓
```

## 🎯 Benefits

- ✅ **No backend code changes needed**
- ✅ **No manual module installation**
- ✅ **No server restarts required**
- ✅ **requirements.txt auto-updated**
- ✅ **Works with any PyPI package**

## 📋 Files Added

- `backend/module_manager.py` - Core auto-install logic
- `backend/restart_server.sh` - Server restart script (if needed)
- `backend/AUTO_MODULE_INSTALL.md` - Full documentation
- `backend/test_module_manager.py` - Test suite

## 📝 Requirements.txt Management

The system automatically updates `requirements.txt` when new modules are installed. To review what's been added:

```bash
cat backend/requirements.txt
```

## 🐛 Troubleshooting

### Tool fails with "No module named X"

**Usually**: The system will auto-install and retry. You'll see a brief delay, then success.

**If it persists**: 
- Check the backend logs: `tail -f /tmp/backend.log`
- The module might need system dependencies (like tesseract-ocr for pytesseract)

### Module installs but still doesn't work

Some modules need **system packages**:

```bash
# Example: tesseract for pytesseract
sudo apt-get install tesseract-ocr

# Example: system libraries for opencv
sudo apt-get install libgl1-mesa-glx
```

### Want to manually restart server?

```bash
cd /home/seehorn/ProgramAT/backend
./restart_server.sh
```

## 🔒 Security Notes

- Only Python packages from PyPI can be auto-installed
- No sudo/root privileges required
- Installations happen in the virtual environment only
- All installations are logged

## 📚 Examples

### Example 1: Text Recognition
```python
import pytesseract  # Pre-loaded ✓

def main(image, input_data):
    return pytesseract.image_to_string(image)
```
**First run**: Instant (already loaded)

### Example 2: Advanced Image Processing
```python
from skimage import filters  # Not pre-loaded

def main(image, input_data):
    edges = filters.sobel(image)
    return "Edge detection complete"
```
**First run**: 3-5 seconds (auto-installs scikit-image)
**Second run**: Instant (cached)

### Example 3: Data Analysis
```python
import pandas as pd  # Not pre-loaded
import numpy as np   # Pre-loaded ✓

def main(image, input_data):
    data = pd.DataFrame({'values': [1, 2, 3]})
    return f"Analysis: {data.mean()}"
```
**First run**: 3-5 seconds (auto-installs pandas)
**Second run**: Instant (cached)

## ⚙️ Advanced Configuration

See `backend/AUTO_MODULE_INSTALL.md` for:
- Adding custom pre-loaded modules
- Disabling auto-install
- Security considerations
- Performance tuning

## 🎓 Development Workflow

1. Write tool code with any imports you need
2. Commit to GitHub
3. Create/update issue in ProgramAT repo
4. Tool appears in app automatically
5. First user runs it → modules auto-install
6. All future users have instant access

**No backend deployment needed!** 🎉

## 📞 Support

If you encounter issues:
1. Check backend logs: `tail -f /tmp/backend.log`
2. Try manual install: `pip install <module-name>`
3. Check for system dependencies: `apt-cache search <module-name>`

## 🚀 Next Steps

Your backend is running with auto-install enabled. Try running a tool that imports a new module and watch it install automatically!

Current backend status: ✅ Running on port 8080
Module manager: ✅ Active
Auto-install: ✅ Enabled
