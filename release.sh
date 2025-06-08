#!/bin/bash
#
# Usage:
# 1. Make sure the script is executable: chmod +x release.sh
# 2. Run the script: ./release.sh
#
# This script:
# - Updates requirements.txt
# - Commits any uncommitted changes automatically before version bump
# - Prompts for version bump (patch/minor/major)
# - Bumps version, updates changelog
# - Builds and optionally uploads to PyPI
# - Commits changelog and pushes commits and tags to GitHub
#
# Make sure you are in the repository root and have your environment configured.

# Exit on any error
set -e

# === STEP 0: Commit uncommitted changes if any ===
if ! git diff-index --quiet HEAD --; then
  echo "Uncommitted changes detected. Auto-committing before version bump..."
  git add .
  git commit -m "Auto-commit: save work before version bump"
else
  echo "No uncommitted changes detected."
fi

# === STEP 1: Update requirements.txt ===
echo "Updating requirements.txt..."
pipreqs . --force
echo "requirements.txt updated successfully."

# === STEP 3: Update FUNCTIONS ===

python3 - <<EOF
import os
from eppink_utilities.file_utils import markdown_append
from eppink_utilities.general_utils import extract_functions_with_docstrings, list_py_files

source_dir = 'eppink_utilities'
output_file = 'FUNCTIONS.md'
output_dir = '.'

py_files = list_py_files(source_dir)

# Clear old file
with open(os.path.join(output_dir, output_file), 'w', encoding='utf-8') as f:
    f.write('# Function List\n\n')

for file in py_files:
    functions = extract_functions_with_docstrings(file)
    header = f"### From \`{os.path.basename(file)}\`\n\n"
    markdown_append([header] + functions, output_file, output_dir)

print(f"Documentation written to {output_file}")
EOF

# === STEP 4: Version bump ===
echo "Which version bump? (patch / minor / major)"
read BUMP
bumpversion "$BUMP"
VERSION=$(grep version pyproject.toml | head -1 | cut -d '"' -f2)
DATE=$(date +"%Y-%m-%d")

# Update version and release date in CITATION.cff
sed -E "s/^(version: \").*(\")/\1$VERSION\2/" CITATION.cff
sed -E "s/^(date-released: ).*/\1$DATE/" CITATION.cff


# === STEP 5: Append to CHANGELOG.md ===
echo -e "\n## Version $VERSION \n$DATE\n### Changed\n- Describe your changes here" >> CHANGELOG.md
echo "Changelog updated. Please edit CHANGELOG.md to describe your changes."

# === STEP 6: Rebuild the package ===
echo "Cleaning and rebuilding package..."
rm -rf dist
python -m build

# === STEP 7: Upload to PyPI ===
echo "Ready to upload to PyPI? (y/n)"
read UPLOAD
if [[ "$UPLOAD" == "y" ]]; then
  twine upload dist/*
else
  echo "Skipped upload."
fi

# === STEP 8: Commit and push changes to GitHub automatically ===
echo "Committing and pushing changelog and tags to GitHub..."

git add CHANGELOG.md

git commit -m "Release version $VERSION"

git push

git push --tags

echo "GitHub updated successfully."
