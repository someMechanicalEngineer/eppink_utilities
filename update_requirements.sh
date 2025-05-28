#!/bin/bash

# Update requirements.txt using pipreqs
echo "Updating requirements.txt..."
pipreqs . --force

echo "requirements.txt updated successfully."

# In order to run this code:
# Run this command in your terminal (while in your project folder):
# chmod +x update_requirements.sh 