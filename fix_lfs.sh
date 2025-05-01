#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Fixing Git history for LFS tracking of .parquet files ---"

# --- Step 1: Undo the last commit but keep changes staged ---
echo "[1/8] Undoing the last commit (soft reset)..."
git reset --soft HEAD~1
echo "    Last commit undone. Files remain staged."

# --- Step 2: Ensure Git LFS is initialized for this repository ---
# This installs the necessary Git hooks for LFS in the current repo.
# It's safe to run even if already installed.
echo "[2/8] Initializing Git LFS for the repository (if needed)..."
git lfs install
echo "    Git LFS initialized."

# --- Step 3: Tell Git LFS to track *.parquet files ---
# This command creates or modifies the .gitattributes file.
echo "[3/8] Configuring LFS to track '*.parquet' files..."
git lfs track "*.parquet"
echo "    '.gitattributes' created or updated."

# --- Step 4: Stage the .gitattributes file ---
# The configuration change needs to be part of the commit.
echo "[4/8] Staging the '.gitattributes' file..."
git add .gitattributes
echo "    '.gitattributes' staged."

# --- Step 5: Unstage the parquet files ---
# This is essential so they can be correctly processed by LFS hooks in the next step.
# We assume they match the pattern "*.parquet". Adjust if paths are complex.
echo "[5/8] Unstaging existing '*.parquet' files..."
# Use --ignore-unmatch to avoid errors if no parquet files were staged (e.g., if reset removed them)
# The '*' ensures it catches files in subdirectories too.
git reset HEAD -- "*.parquet" || echo "    No staged parquet files needed unstaging (or potential error)."
echo "    Parquet files unstaged."

# --- Step 6: Re-stage all files ---
# LFS hooks will now intercept the .parquet files and replace them with pointers.
# Using 'git add .' is generally recommended over 'git add *' as it correctly
# handles filenames starting with '.' and respects .gitignore rules better.
echo "[6/8] Re-staging all files (LFS will process '*.parquet')..."
git add .
echo "    All files re-staged. LFS should now handle parquet files."

# --- Step 7: Commit the changes ---
# Using the commit message you specified.
echo "[7/8] Creating the corrected commit..."
git commit -m "initial commit (fixed)"
echo "    Commit created successfully."

# --- Step 8: Push the corrected commit ---
# This assumes your remote is named 'origin' and you are pushing the current branch.
# Adjust 'origin' and add branch name if needed (e.g., git push origin main)
echo "[8/8] Pushing the commit to the remote repository..."
git push
echo "    Push command executed."

echo "--- Process Complete ---"
echo "The last commit has been modified to use Git LFS for '.parquet' files."
echo "Please verify on your remote repository (e.g., GitHub, GitLab) that the"
echo "'.parquet' files are now listed as LFS pointers (usually showing their size)."
echo "Check 'git status' and 'git lfs ls-files' to ensure everything looks correct locally."

