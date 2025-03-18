import os
import importlib.util

# Get the directory containing this file
_this_dir = os.path.dirname(os.path.abspath(__file__))

# Find the extension file (look for _core*.so or similar)
for filename in os.listdir(_this_dir):
    if filename.startswith('_core') and (filename.endswith('.so') or
                                         filename.endswith('.pyd') or
                                         filename.endswith('.dylib')):
        # Full path to the extension
        extension_path = os.path.join(_this_dir, filename)

        # Load the extension module directly from file path
        spec = importlib.util.spec_from_file_location("_temp_module", extension_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Copy all symbols from the extension module to this module's namespace
        for name in dir(module):
            if not name.startswith('_'):
                # noinspection PyInterpreter
                globals()[name] = getattr(module, name)

        # Extension successfully loaded
        break
else:
    # No extension file found
    raise ImportError(f"No extension module found in {_this_dir}")