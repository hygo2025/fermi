#!/bin/bash
# Setup script for session-rec wrappers
# Creates symlink from session-rec-lib/algorithms/models to src/models

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "Setting up session-rec wrappers..."

# Check if session-rec-lib exists
if [ ! -d "session-rec-lib" ]; then
    echo "Error: session-rec-lib not found. Run 'make install-benchmark' first."
    exit 1
fi

# Create symlink
SYMLINK_PATH="session-rec-lib/algorithms/models"
TARGET_PATH="../../src/models"

if [ -L "$SYMLINK_PATH" ]; then
    echo "✓ Symlink already exists: $SYMLINK_PATH -> $TARGET_PATH"
elif [ -e "$SYMLINK_PATH" ]; then
    echo "Warning: $SYMLINK_PATH exists but is not a symlink. Removing..."
    rm -rf "$SYMLINK_PATH"
    ln -sf "$TARGET_PATH" "$SYMLINK_PATH"
    echo "✓ Symlink created: $SYMLINK_PATH -> $TARGET_PATH"
else
    ln -sf "$TARGET_PATH" "$SYMLINK_PATH"
    echo "✓ Symlink created: $SYMLINK_PATH -> $TARGET_PATH"
fi

# Verify wrappers exist
echo ""
echo "Checking wrappers..."
if [ -f "src/models/knn/iknn.py" ]; then
    echo "  ✓ IKNN wrapper found"
else
    echo "  ✗ IKNN wrapper missing"
fi

if [ -f "src/models/knn/sknn.py" ]; then
    echo "  ✓ SKNN wrapper found"
else
    echo "  ✗ SKNN wrapper missing"
fi

echo ""
echo "✓ Setup complete!"
