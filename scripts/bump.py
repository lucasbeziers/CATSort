import sys
import re
from pathlib import Path
import subprocess

def bump(new_version):
    # Ensure version format is correct (basic check)
    if not re.match(r"^\d+\.\d+\.\d+$", new_version):
        print(f"Error: Version '{new_version}' should be in format X.Y.Z")
        sys.exit(1)

    # 1. Update pyproject.toml
    pyproject = Path("pyproject.toml")
    if pyproject.exists():
        content = pyproject.read_text(encoding="utf-8")
        new_content = re.sub(r'version = "[^"]+"', f'version = "{new_version}"', content, count=1)
        pyproject.write_text(new_content, encoding="utf-8")
        print(f"Updated {pyproject}")

    # 2. Update __init__.py
    init_py = Path("src/catsort/__init__.py")
    if init_py.exists():
        content = init_py.read_text(encoding="utf-8")
        new_content = re.sub(r'__version__ = "[^"]+"', f'__version__ = "{new_version}"', content, count=1)
        init_py.write_text(new_content, encoding="utf-8")
        print(f"Updated {init_py}")

    # 3. Git Operations
    try:
        subprocess.run(["git", "add", "pyproject.toml", "src/catsort/__init__.py"], check=True)
        subprocess.run(["git", "commit", "-m", f"chore: bump version to {new_version}"], check=True)
        subprocess.run(["git", "tag", "-a", f"{new_version}", "-m", f"Release {new_version}"], check=True)
        print(f"\nSuccessfully bumped to {new_version}!")
        print(f"Now run: git push origin --description --tags")
    except subprocess.CalledProcessError as e:
        print(f"Error during git operations: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/bump.py <new_version>")
    else:
        bump(sys.argv[1])
