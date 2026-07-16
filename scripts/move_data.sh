#!/bin/bash
set -euo pipefail

# Directories
SRC_DIR="/home/robertorojas/Precipitation/data"
DEST_DIR="/mnt/data-r2/RobertoRojas/downscaling"

echo "=== Data Migration Script ==="
echo "Source: $SRC_DIR"
echo "Destination: $DEST_DIR"
echo ""

# 1. Check if source exists
if [ ! -d "$SRC_DIR" ]; then
    echo "Error: Source directory $SRC_DIR does not exist!"
    exit 1
fi

# 2. Check source size
SRC_SIZE=$(du -sh "$SRC_DIR" 2>/dev/null | cut -f1 || echo "unknown")
echo "Source directory size: $SRC_SIZE"

# 3. Create destination directory if it doesn't exist
echo "Creating destination directory if it doesn't exist..."
mkdir -p "$DEST_DIR"

# 4. Perform migration using rsync
echo "Starting file migration via rsync..."
echo "This may take a long time. Successfully migrated files will be removed from source."
# --remove-source-files: deletes source files after successful transfer
# --inplace: writes directly to target files (avoiding mkstemp rename issues)
# --no-perms, --no-owner, --no-group, --no-times: avoids setting permissions/owners/times that are not permitted on the mount

MAX_RETRIES=50
RETRY_COUNT=0
DELAY=15

while true; do
    if rsync -rlD --inplace --no-perms --no-owner --no-group --no-times --remove-source-files -v "$SRC_DIR/" "$DEST_DIR/"; then
        echo "rsync migration completed successfully."
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
            echo "Error: rsync failed after $MAX_RETRIES attempts."
            exit 1
        fi
        echo "rsync failed (attempt $RETRY_COUNT/$MAX_RETRIES). Destination mount may have dropped. Checking connection..."
        # Wait until the destination directory is readable/accessible again
        while [ ! -d "$DEST_DIR" ]; do
            echo "Destination directory $DEST_DIR is inaccessible (Host is down?). Waiting for host/mount to recover..."
            sleep 10
        done
        echo "Destination directory is accessible. Retrying rsync in $DELAY seconds..."
        sleep $DELAY
    fi
done

# 5. Clean up remaining empty directories in source
echo "Cleaning up empty directories in source..."
find "$SRC_DIR" -type d -empty -delete || true

# 6. Check if source is completely empty or can be removed
if [ -d "$SRC_DIR" ]; then
    if [ -z "$(ls -A "$SRC_DIR")" ]; then
        echo "Removing empty source directory $SRC_DIR..."
        rmdir "$SRC_DIR"
    else
        echo "Warning: Source directory $SRC_DIR is not empty. Some files/directories may not have been moved."
        echo "Remaining files/directories:"
        ls -la "$SRC_DIR"
    fi
fi

# 7. Check destination size
DEST_SIZE=$(du -sh "$DEST_DIR" 2>/dev/null | cut -f1 || echo "unknown")
echo "Migration complete!"
echo "Destination directory size: $DEST_SIZE"
