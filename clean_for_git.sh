#!/bin/bash

# Script to clean up large files before committing to git
# Run this script from the project root directory

echo "🧹 Cleaning up large files for git repository..."

# Create a backup directory for large files (outside git)
BACKUP_DIR="../job-market-intelligence-large-files"
mkdir -p "$BACKUP_DIR"

echo "📁 Created backup directory: $BACKUP_DIR"

# Move large model files
echo "🤖 Moving model files..."
find models/src/generation/model_cache/ -name "*.pt" -o -name "*.pth" 2>/dev/null | while read file; do
    if [ -f "$file" ]; then
        echo "  Moving: $file"
        mkdir -p "$BACKUP_DIR/$(dirname "$file")"
        mv "$file" "$BACKUP_DIR/$file"
    fi
done

# Move large data files  
echo "📊 Moving large data files..."
find models/src/generation/data/processed/ -name "*.parquet" -o -name "*.npy" 2>/dev/null | while read file; do
    if [ -f "$file" ]; then
        echo "  Moving: $file"
        mkdir -p "$BACKUP_DIR/$(dirname "$file")"
        mv "$file" "$BACKUP_DIR/$file"
    fi
done

# Move visualization GIFs
echo "🎬 Moving visualization files..."
find models/src/generation/ -name "*.gif" 2>/dev/null | while read file; do
    if [ -f "$file" ]; then
        echo "  Moving: $file"
        mkdir -p "$BACKUP_DIR/$(dirname "$file")"
        mv "$file" "$BACKUP_DIR/$file"
    fi
done

# Move pickle files
echo "🥒 Moving pickle files..."
find models/src/generation/ -name "*.pkl" 2>/dev/null | while read file; do
    if [ -f "$file" ]; then
        echo "  Moving: $file"
        mkdir -p "$BACKUP_DIR/$(dirname "$file")"
        mv "$file" "$BACKUP_DIR/$file"
    fi
done

echo "✅ Cleanup complete!"
echo "📋 Large files have been moved to: $BACKUP_DIR"
echo "💡 To restore files after cloning, copy them back from the backup directory"
echo ""
echo "🚀 Your repository is now ready for git!"
echo "   Run: git add . && git commit -m 'Clean codebase for GitHub'"