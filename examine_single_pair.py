#!/usr/bin/env python
# coding: utf-8

"""
Script to examine a single pair of PSG and hypnogram files
"""

import os
import sys
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Specify the files to examine
PSG_FILE = "ST7011J0-PSG.edf"
HYPNO_FILE = "ST7011JP-Hypnogram.edf"

# Output file
OUTPUT_FILE = "single_pair_examination.txt"

def main():
    """Main function to examine a single pair of files"""
    with open(OUTPUT_FILE, 'w') as f:
        f.write(f"Examining PSG file: {PSG_FILE}\n")
        f.write(f"Examining hypnogram file: {HYPNO_FILE}\n\n")
        
        # Check if files exist
        if not os.path.exists(PSG_FILE):
            f.write(f"Error: PSG file {PSG_FILE} not found.\n")
            return
        
        if not os.path.exists(HYPNO_FILE):
            f.write(f"Error: Hypnogram file {HYPNO_FILE} not found.\n")
            return
        
        # Examine PSG file
        try:
            f.write("=== PSG File Analysis ===\n")
            raw = mne.io.read_raw_edf(PSG_FILE, preload=True)
            
            f.write(f"Channels: {raw.ch_names}\n")
            f.write(f"Number of channels: {len(raw.ch_names)}\n")
            f.write(f"Channel types: {raw.get_channel_types()}\n")
            f.write(f"Sampling rate: {raw.info['sfreq']} Hz\n")
            f.write(f"Duration: {raw.times[-1]:.2f} seconds ({raw.times[-1]/60:.2f} minutes)\n")
            f.write(f"Number of data points: {len(raw.times)}\n\n")
            
            # Get data for the first 5 seconds of the first channel
            data, times = raw[0, :int(5 * raw.info['sfreq'])]
            f.write(f"First 10 data points of first channel: {data[0][:10]}\n\n")
            
            # Save data to numpy file for later use
            np.save('psg_data_sample.npy', raw.get_data())
            f.write("PSG data saved to 'psg_data_sample.npy'\n")
            
            # Plot and save figure
            f.write("Creating PSG plot...\n")
            fig = plt.figure(figsize=(12, 8))
            for i, ch_name in enumerate(raw.ch_names[:4]):  # Plot first 4 channels
                plt.subplot(4, 1, i+1)
                plt.plot(raw.times[:int(30 * raw.info['sfreq'])], 
                        raw.get_data()[i, :int(30 * raw.info['sfreq'])])
                plt.title(ch_name)
                if i == 3:  # Only add x-label to bottom subplot
                    plt.xlabel('Time (s)')
            plt.tight_layout()
            plt.savefig('psg_sample.png')
            f.write("PSG plot saved as 'psg_sample.png'\n\n")
            
        except Exception as e:
            f.write(f"Error examining PSG file: {str(e)}\n")
        
        # Examine hypnogram file
        try:
            f.write("=== Hypnogram File Analysis ===\n")
            annot = mne.read_annotations(HYPNO_FILE)
            
            f.write(f"Number of annotations: {len(annot)}\n")
            
            if len(annot) > 0:
                f.write("First 10 annotations:\n")
                for i in range(min(10, len(annot))):
                    f.write(f"  {i}: Onset={annot[i]['onset']:.2f}s, Duration={annot[i]['duration']:.2f}s, Description={annot[i]['description']}\n")
                
                # Count unique sleep stages
                stages = {}
                for a in annot:
                    stage = a['description']
                    if stage in stages:
                        stages[stage] += 1
                    else:
                        stages[stage] = 1
                
                f.write("\nSleep stage distribution:\n")
                for stage, count in stages.items():
                    f.write(f"  {stage}: {count} epochs ({count * 30 / 60:.2f} minutes)\n")
                
                # Convert annotations to a hypnogram array
                stage_map = {
                    'Sleep stage W': 0,
                    'Sleep stage 1': 1,
                    'Sleep stage 2': 2,
                    'Sleep stage 3': 3,
                    'Sleep stage 4': 3,  # Combine stage 3 and 4 as N3
                    'Sleep stage R': 4,
                    'Movement time': -1,
                    'Sleep stage ?': -1
                }
                
                # Determine the total duration
                total_duration = max([a['onset'] + a['duration'] for a in annot])
                epoch_length = 30  # Standard 30-second epochs
                n_epochs = int(total_duration / epoch_length) + 1
                
                # Initialize hypnogram array
                hypnogram = np.zeros(n_epochs) - 1  # -1 for unknown
                
                # Fill in the hypnogram
                for a in annot:
                    onset_epoch = int(a['onset'] / epoch_length)
                    duration_epochs = int(a['duration'] / epoch_length)
                    if a['description'] in stage_map:
                        stage = stage_map[a['description']]
                        for i in range(duration_epochs):
                            if onset_epoch + i < n_epochs:
                                hypnogram[onset_epoch + i] = stage
                
                # Save hypnogram data
                np.save('hypnogram_data.npy', hypnogram)
                f.write("\nHypnogram data saved to 'hypnogram_data.npy'\n")
                
                # Plot the hypnogram
                f.write("Creating hypnogram plot...\n")
                plt.figure(figsize=(12, 6))
                plt.plot(np.arange(n_epochs) * epoch_length / 60, hypnogram, 'b-')
                plt.yticks([-1, 0, 1, 2, 3, 4], ['Unknown', 'Wake', 'N1', 'N2', 'N3', 'REM'])
                plt.title(f"Hypnogram from {HYPNO_FILE}")
                plt.xlabel('Time (minutes)')
                plt.ylabel('Sleep Stage')
                plt.grid(True)
                plt.savefig('hypnogram_sample.png')
                f.write("Hypnogram plot saved as 'hypnogram_sample.png'\n")
            
        except Exception as e:
            f.write(f"Error examining hypnogram file: {str(e)}\n")
        
        f.write("\nExamination complete.\n")
        print(f"Examination results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
