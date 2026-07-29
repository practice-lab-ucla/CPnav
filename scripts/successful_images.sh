#!/bin/bash

# Check if the user provided a directory argument
if [ -z "$1" ]; then
    echo "Usage: $0 <root_directory>"
    exit 1
fi

ROOT_DIR="$1"
TARGET_DIR="$ROOT_DIR/goal_image"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "Processing episodes in: $ROOT_DIR"
echo "Images will be saved to: $TARGET_DIR"

# Find and process the images
find "$ROOT_DIR" -maxdepth 4 -path "*/terminal/*.png" -exec bash -c '
    filepath="$1"
    target_dir="$2"
    
    # Extract goal name (filename without .png)
    goal=$(basename "$filepath" .png)
    
    # Get the episode folder (e.g., episode_42_timestamp)
    # dirname $filepath gives .../terminal/
    # dirname $(dirname $filepath) gives .../episode_42_timestamp/
    episode_folder=$(basename "$(dirname "$(dirname "$filepath")")")
    
    # Extract the number (the part after "episode_")
    # This splits by "_" and takes the 2nd field
    ep_num=$(echo "$episode_folder" | cut -d"_" -f2)
    
    # Perform the copy and rename
    cp "$filepath" "$target_dir/${ep_num}_${goal}.png"
' _ {} "$TARGET_DIR" \;

echo "Done! Check $TARGET_DIR for your renamed images."