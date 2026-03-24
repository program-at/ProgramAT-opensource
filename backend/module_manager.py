"""
Module Manager for Automatic Dependency Installation
Handles dynamic module loading and installation without backend code changes
"""

import subprocess
import sys
import importlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModuleManager:
    """Manages automatic installation and loading of Python modules"""
    
    # Map of import names to pip package names
    PACKAGE_NAME_MAP = {
        'cv2': 'opencv-python',
        'PIL': 'pillow',
        'sklearn': 'scikit-learn',
        'yaml': 'pyyaml',
        'dotenv': 'python-dotenv',
    }
    
    # Reverse map: pip package name to import name
    IMPORT_NAME_MAP = {v: k for k, v in PACKAGE_NAME_MAP.items()}
    
    def __init__(self, requirements_file='requirements.txt'):
        self.requirements_file = Path(__file__).parent / requirements_file
        self.installed_modules = set()
        self.module_cache = {}
        
    def install_module(self, module_name):
        """Install a module using pip"""
        # Check if already installed in this session
        if module_name in self.installed_modules:
            logger.info(f"Module {module_name} already installed in this session, skipping")
            return True
            
        try:
            logger.info(f"Installing module: {module_name}")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", module_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.installed_modules.add(module_name)
            
            # Invalidate import cache for this module
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # Also invalidate cache for submodules
            keys_to_delete = [key for key in sys.modules.keys() if key.startswith(module_name + '.')]
            for key in keys_to_delete:
                del sys.modules[key]
            
            # Update requirements.txt if not already there
            self._update_requirements(module_name)
            
            logger.info(f"Successfully installed: {module_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {module_name}: {e}")
            return False
    
    def _update_requirements(self, module_name):
        """Add module to requirements.txt if not already present"""
        try:
            # Read existing requirements
            existing = set()
            if self.requirements_file.exists():
                with open(self.requirements_file, 'r') as f:
                    existing = {line.split('>=')[0].split('==')[0].strip() 
                               for line in f if line.strip() and not line.startswith('#')}
            
            # Add new module if not present
            if module_name not in existing:
                with open(self.requirements_file, 'a') as f:
                    f.write(f"\n{module_name}\n")
                logger.info(f"Added {module_name} to requirements.txt")
        except Exception as e:
            logger.warning(f"Could not update requirements.txt: {e}")
    
    def load_module(self, module_name, install_if_missing=True):
        """Load a module, optionally installing it if missing"""
        # Check cache first
        if module_name in self.module_cache:
            return self.module_cache[module_name]
        
        try:
            # Try to import the module
            module = importlib.import_module(module_name)
            self.module_cache[module_name] = module
            return module
        except ImportError:
            if install_if_missing and module_name not in self.installed_modules:
                logger.info(f"Module {module_name} not found, attempting installation...")
                
                # Get the correct pip package name
                package_name = self.PACKAGE_NAME_MAP.get(module_name, module_name)
                
                if self.install_module(package_name):
                    try:
                        # Clear our cache to force fresh import
                        if module_name in self.module_cache:
                            del self.module_cache[module_name]
                        
                        # Try importing again after installation (use import name, not package name)
                        importlib.invalidate_caches()  # Invalidate Python's import cache
                        module = importlib.import_module(module_name)
                        self.module_cache[module_name] = module
                        return module
                    except ImportError as e:
                        logger.error(f"Failed to import {module_name} after installation: {e}")
                        return None
                else:
                    logger.error(f"Installation of {package_name} failed")
                    return None
            return None
    
    def get_common_modules(self):
        """Get commonly used modules for tool execution"""
        # Import typing components separately so they can be used directly
        import typing
        
        common = {
            'cv2': self.load_module('cv2'),
            'np': self.load_module('numpy'),
            'Image': None,  # Will be set from PIL.Image
            'datetime': self.load_module('datetime'),
            'pytesseract': self.load_module('pytesseract'),
            'scipy': self.load_module('scipy'),
            'sklearn': self.load_module('sklearn'),  # Import name is sklearn, package is scikit-learn
            're': self.load_module('re'),
            'json': self.load_module('json'),
            'math': self.load_module('math'),
            'typing': self.load_module('typing'),  # For type hints (Optional, List, Dict, etc.)
            # Also include commonly used typing components directly
            'Optional': typing.Optional,
            'List': typing.List,
            'Dict': typing.Dict,
            'Tuple': typing.Tuple,
            'Union': typing.Union,
            'Any': typing.Any,
        }
        
        # Handle PIL.Image specially
        try:
            from PIL import Image
            common['Image'] = Image
        except ImportError:
            if self.install_module('pillow'):
                from PIL import Image
                common['Image'] = Image
        
        # Filter out None values (failed imports)
        return {k: v for k, v in common.items() if v is not None}
    
    def install_from_error(self, error_message):
        """
        Parse an ImportError/ModuleNotFoundError and install the missing module
        Returns the module name if installation was attempted, None otherwise
        """
        # Common patterns for module errors
        patterns = [
            "No module named '",
            "No module named ",
            "cannot import name '",
        ]
        
        for pattern in patterns:
            if pattern in error_message:
                # Extract module name
                start = error_message.find(pattern) + len(pattern)
                end = error_message.find("'", start)
                if end == -1:
                    # No closing quote, try to find space or end
                    parts = error_message[start:].split()
                    module_name = parts[0] if parts else None
                else:
                    module_name = error_message[start:end]
                
                if module_name:
                    # Handle submodule imports (e.g., 'scipy.ndimage' -> 'scipy')
                    base_module = module_name.split('.')[0]
                    
                    # Get the correct pip package name
                    package_name = self.PACKAGE_NAME_MAP.get(base_module, base_module)
                    
                    # Check if we already tried to install this package
                    if package_name in self.installed_modules:
                        logger.warning(f"Package {package_name} (for {base_module}) already installed but still failing to import")
                        return None  # Don't retry installation
                    
                    # Install the package
                    if self.install_module(package_name):
                        return base_module  # Return import name, not package name
        
        return None

# Global module manager instance
_module_manager = None

def get_module_manager():
    """Get or create the global module manager instance"""
    global _module_manager
    if _module_manager is None:
        _module_manager = ModuleManager()
    return _module_manager
