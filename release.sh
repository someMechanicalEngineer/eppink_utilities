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

# === STEP 2: Ask for version bump type ===
echo "Which version bump? (patch / minor / major)"
read BUMP

# === STEP 3: Bump the version and tag it ===
bump2version "$BUMP"

# === STEP 4: Get new version and date ===
VERSION=$(grep version pyproject.toml | head -1 | cut -d '"' -f2)
DATE=$(date +"%Y-%m-%d")

# === STEP 5: Append to CHANGELOG.md ===
echo -e "\n## Version $VERSION\n$DATE\n### Changed\n- Describe your changes here" >> CHANGELOG.md
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
