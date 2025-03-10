#!/usr/bin/env python
# coding: utf-8

"""
Simple script to list files in the current directory
"""

import os

# Get all files in the current directory
files = os.listdir('.')

# Open a file to save the output
with open('file_list.txt', 'w') as f:
    f.write(f"Total files: {len(files)}\n\n")
    
    # Count EDF files
    psg_files = [file for file in files if file.endswith('-PSG.edf')]
    hypno_files = [file for file in files if file.endswith('-Hypnogram.edf')]
    
    f.write(f"PSG files: {len(psg_files)}\n")
    f.write(f"Hypnogram files: {len(hypno_files)}\n\n")
    
    # List all files
    f.write("All files:\n")
    for file in sorted(files):
        f.write(f"- {file}\n")

print(f"File list saved to file_list.txt")
