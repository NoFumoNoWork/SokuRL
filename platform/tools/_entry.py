"""Preserve old CLI names as references to the single packaged implementation."""
import importlib
from pathlib import Path
import runpy
import sys


def expose(name, namespace):
    source = str(Path(__file__).resolve().parents[1] / 'src')
    if source not in sys.path:
        sys.path.insert(0, source)
    qualified = 'sokurl_platform.' + name
    if namespace['__name__'] == '__main__':
        runpy.run_module(qualified, run_name='__main__')
    else:
        module = importlib.import_module(qualified)
        namespace.update({key: value for key, value in vars(module).items()
                          if not key.startswith('__')})
        sys.modules[namespace['__name__']] = module
