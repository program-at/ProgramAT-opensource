# Automatic Module Installation System

## Overview

The backend now includes an **automatic module installation system** that detects missing Python modules when tools execute and installs them on-the-fly without requiring manual backend code changes.

## How It Works

### 1. Module Manager (`module_manager.py`)

The `ModuleManager` class provides:
- **Automatic detection** of missing modules from error messages
- **Automatic installation** using pip
- **Automatic updating** of `requirements.txt`
- **Module caching** to avoid repeated imports
- **Pre-loading** of common modules (cv2, numpy, pytesseract, scipy, etc.)

### 2. Tool Execution with Auto-Install

When a tool is executed in `stream_server.py`:

1. **Initial Execution**: The tool code runs with pre-loaded common modules
2. **Error Detection**: If an `ImportError` or `ModuleNotFoundError` occurs:
   - The error message is parsed to extract the missing module name
   - The module is automatically installed via pip
   - The module is added to `requirements.txt`
3. **Retry**: The tool execution is retried with the newly installed module
4. **Success**: The tool runs successfully with all required modules

### 3. Retry Logic

```python
max_retries = 3
retry_count = 0

while retry_count < max_retries:
    try:
        exec(tool_code, exec_globals, exec_locals)
        break  # Success
    except (ImportError, ModuleNotFoundError) as e:
        # Auto-install missing module
        installed_module = module_mgr.install_from_error(str(e))
        if installed_module:
            # Reload modules and retry
            retry_count += 1
        else:
            raise  # Can't fix, re-raise error
```

## Pre-Loaded Modules

The following modules are pre-loaded and available to all tools:

- `cv2` - OpenCV for computer vision
- `np` - NumPy for numerical operations
- `Image` - PIL/Pillow for image processing
- `datetime` - Date and time operations
- `pytesseract` - OCR capabilities
- `scipy` - Scientific computing
- `sklearn` - Machine learning (scikit-learn)
- `re` - Regular expressions
- `json` - JSON parsing
- `math` - Mathematical functions

## Benefits

### No Backend Code Changes Required

- ✅ Tools can import any Python module they need
- ✅ Modules are installed automatically on first use
- ✅ No need to manually edit `stream_server.py`
- ✅ No need to restart the server manually
- ✅ `requirements.txt` is automatically updated

### Dynamic Dependency Management

- New tools with new dependencies "just work"
- Dependencies are discovered at runtime
- All installations are tracked in `requirements.txt`

### Backwards Compatible

- Existing tools continue to work exactly as before
- Pre-loaded modules are immediately available
- No migration required

## Example Usage

### Tool That Uses scikit-image (Not Pre-Loaded)

```python
# tools/edge_detector.py
from skimage import filters, color

def main(image, input_data):
    # Convert to grayscale
    gray = color.rgb2gray(image)
    
    # Detect edges
    edges = filters.sobel(gray)
    
    return f"Edge detection complete"
```

**What Happens:**
1. First execution: `skimage` not found → auto-install → retry → success
2. Subsequent executions: `skimage` is cached → instant success
3. `requirements.txt` now contains: `scikit-image`

### Tool That Uses Pre-Loaded Module

```python
# tools/text_reader.py
import pytesseract

def main(image, input_data):
    text = pytesseract.image_to_string(image)
    return text
```

**What Happens:**
1. First execution: `pytesseract` already loaded → instant success
2. No installation needed

## Module Installation Process

When a missing module is detected:

1. **Parse Error**: Extract module name from error message
   - `"No module named 'scipy'"` → `scipy`
   - `"No module named 'sklearn'"` → `sklearn`
   - `"cannot import name 'filters'"` → check submodule

2. **Install**: Run `pip install <module_name>`
   - Silent installation (no output shown)
   - Uses the virtual environment's pip

3. **Update Requirements**: Append to `requirements.txt`
   - Only if not already present
   - Prevents duplicates

4. **Reload & Cache**: Import and cache the new module
   - Available immediately for retry
   - Cached for future use

## Limitations

### Retry Limit
- Maximum 3 retry attempts per tool execution
- Prevents infinite loops on persistent errors

### System Dependencies
- Only Python modules can be auto-installed
- System packages (like `tesseract-ocr`) still need manual installation
- C libraries or compiled dependencies may require apt-get

### Installation Failures
- If pip fails to install a module, the error is returned to the client
- User will see the original import error
- May need manual intervention for complex dependencies

## Restart Script

The `restart_server.sh` script can be used to manually restart the backend:

```bash
cd /home/seehorn/ProgramAT/backend
./restart_server.sh
```

This script:
- Kills existing server process
- Activates virtual environment
- Starts server in background
- Logs to `/tmp/backend.log`

## Troubleshooting

### Module Still Not Found After Installation

**Cause**: Module name mismatch (e.g., `sklearn` vs `scikit-learn`)

**Solution**: The module manager tries to handle common aliases, but some may need manual addition to the common modules list.

### Installation Fails

**Cause**: Network issues, package not available, or requires compilation

**Solution**: 
1. Check `/tmp/backend.log` for pip errors
2. Manually install: `cd backend && source venv/bin/activate && pip install <module>`
3. Add to `requirements.txt`

### Server Not Restarting

**Cause**: Process didn't terminate cleanly

**Solution**:
```bash
pkill -9 -f stream_server.py
cd backend
source venv/bin/activate
python stream_server.py &
```

## Future Enhancements

Potential improvements:

1. **Dependency Resolver**: Handle complex dependency trees
2. **Version Pinning**: Auto-detect and pin versions for reproducibility
3. **Security Sandbox**: Limit which modules can be auto-installed
4. **Installation Cache**: Pre-download common scientific packages
5. **Docker Integration**: Auto-rebuild container on new dependencies

## Configuration

### Adding Pre-Loaded Modules

Edit `module_manager.py`:

```python
def get_common_modules(self):
    common = {
        'cv2': self.load_module('cv2'),
        'np': self.load_module('numpy'),
        # Add your module here:
        'my_module': self.load_module('my-package-name'),
    }
    return {k: v for k, v in common.items() if v is not None}
```

### Disabling Auto-Install

In `stream_server.py`, change:

```python
installed_module = module_mgr.install_from_error(error_msg)
```

To:

```python
installed_module = None  # Disable auto-install
```

## Security Considerations

### Current Approach
- Auto-installation runs in the backend's virtual environment
- Only Python packages from PyPI can be installed
- No privilege escalation (no sudo)

### Production Recommendations
1. Use Docker containers with frozen dependencies
2. Implement allowlist of permitted modules
3. Log all installations for audit trails
4. Set up monitoring for unusual package requests
5. Consider using private PyPI mirror

## Performance

- **First execution**: +2-5 seconds for module installation
- **Subsequent executions**: No overhead (modules are cached)
- **Memory**: Minimal increase from cached modules
- **Storage**: Each module varies (typically 1-50 MB)

## Maintenance

### Regular Updates

Update all installed packages:
```bash
cd backend
source venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Cleanup

Remove unused modules from `requirements.txt`:
```bash
pip-autoremove <package-name>
```

Or use:
```bash
pip uninstall <package-name>
# Then manually remove from requirements.txt
```
