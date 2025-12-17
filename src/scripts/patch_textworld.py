#!/usr/bin/env python3
"""Patch TextWorld for Python 3.13 compatibility.

The locals().update() pattern doesn't work in Python 3.13.
This script patches the textworld library to use eval() with explicit globals instead.
"""

import sys
import site
from pathlib import Path


def find_textworld_textgen():
    """Find the textworld textgen __init__.py file in site-packages."""
    for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
        target = Path(site_dir) / "textworld" / "envs" / "pddl" / "textgen" / "__init__.py"
        if target.exists():
            return target
    
    # Also check the virtual environment
    if hasattr(sys, 'prefix'):
        for python_dir in Path(sys.prefix).glob("lib/python*/site-packages"):
            target = python_dir / "textworld" / "envs" / "pddl" / "textgen" / "__init__.py"
            if target.exists():
                return target
    
    return None


def patch_file(filepath: Path) -> bool:
    """Apply the Python 3.13 compatibility patch."""
    content = filepath.read_text()
    
    old_code = '''    def derive(self, context=None):
        context = context or self.context
        locals().update(context["variables"])
        value = eval(self.expression)
        return [TerminalSymbol(value)]'''
    
    new_code = '''    def derive(self, context=None):
        context = context or self.context
        eval_globals = dict(context["variables"])
        value = eval(self.expression, eval_globals)
        return [TerminalSymbol(value)]'''
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        filepath.write_text(content)
        return True
    elif new_code in content:
        print("Already patched!")
        return True
    else:
        return False


def main():
    print("Patching TextWorld for Python 3.13 compatibility...")
    
    filepath = find_textworld_textgen()
    if filepath is None:
        print("ERROR: Could not find textworld installation.")
        print("Make sure textworld is installed: pip install textworld")
        sys.exit(1)
    
    print(f"Found: {filepath}")
    
    if patch_file(filepath):
        print("SUCCESS: TextWorld patched successfully!")
    else:
        print("ERROR: Could not find the code to patch.")
        print("The file may have a different version or already be modified.")
        sys.exit(1)


if __name__ == "__main__":
    main()
