#!/usr/bin/env python
# coding: utf-8

"""
Simple script to check if files exist and print basic information
"""

import os
import sys

# Files to check
PSG_FILE = "ST7011J0-PSG.edf"
HYPNO_FILE = "ST7011JP-Hypnogram.edf"

def main():
    """Main function to check files"""
    # Check if files exist
    print(f"Checking if {PSG_FILE} exists: {os.path.exists(PSG_FILE)}")
    print(f"Checking if {HYPNO_FILE} exists: {os.path.exists(HYPNO_FILE)}")
    
    # List all files in the current directory
    print("\nFiles in current directory:")
    for file in sorted(os.listdir('.')):
        print(f"- {file}")
    
    # Save output to a file
    with open('check_files_output.txt', 'w') as f:
        f.write(f"Checking if {PSG_FILE} exists: {os.path.exists(PSG_FILE)}\n")
        f.write(f"Checking if {HYPNO_FILE} exists: {os.path.exists(HYPNO_FILE)}\n")
        
        f.write("\nFiles in current directory:\n")
        for file in sorted(os.listdir('.')):
            f.write(f"- {file}\n")
    
    print("\nOutput saved to check_files_output.txt")

if __name__ == "__main__":
    main()
