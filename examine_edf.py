#!/usr/bin/env python
# coding: utf-8

"""
Script to examine EDF files in the current directory
"""

import os
import sys
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Open a file to save the output
output_file = open('edf_examination_results.txt', 'w')

def print_to_file(message):
    """Print message to both console and file"""
    print(message)
    output_file.write(message + '\n')

def examine_psg_file(file_path):
    """Examine a PSG EDF file and print its properties"""
    print_to_file(f"\nExamining PSG file: {file_path}")
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True)
        print_to_file(f"Channels: {raw.ch_names}")
        print_to_file(f"Duration: {raw.times[-1]:.2f} seconds ({raw.times[-1]/60:.2f} minutes)")
        print_to_file(f"Sampling rate: {raw.info['sfreq']} Hz")
        print_to_file(f"Number of data points: {len(raw.times)}")
        return raw
    except Exception as e:
        print_to_file(f"Error reading file: {str(e)}")
        return None

def examine_hypnogram_file(file_path):
    """Examine a hypnogram EDF file and print its properties"""
    print_to_file(f"\nExamining hypnogram file: {file_path}")
    try:
        annot = mne.read_annotations(file_path)
        print_to_file(f"Number of annotations: {len(annot)}")
        if len(annot) > 0:
            print_to_file(f"First few annotations:")
            for i in range(min(5, len(annot))):
                print_to_file(f"  {i}: Onset={annot[i]['onset']:.2f}s, Duration={annot[i]['duration']:.2f}s, Description={annot[i]['description']}")
            
            # Count unique sleep stages
            stages = {}
            for a in annot:
                stage = a['description']
                if stage in stages:
                    stages[stage] += 1
                else:
                    stages[stage] = 1
            
            print_to_file(f"Sleep stage distribution:")
            for stage, count in stages.items():
                print_to_file(f"  {stage}: {count} epochs ({count * 30 / 60:.2f} minutes)")
        
        return annot
    except Exception as e:
        print_to_file(f"Error reading hypnogram: {str(e)}")
        return None

def main():
    """Main function to examine EDF files"""
    # Get all EDF files in the current directory
    edf_files = [f for f in os.listdir('.') if f.endswith('.edf')]
    
    if not edf_files:
        print_to_file("No EDF files found in the current directory.")
        return
    
    print_to_file(f"Found {len(edf_files)} EDF files.")
    
    # Examine one PSG file and its corresponding hypnogram
    psg_files = [f for f in edf_files if f.endswith('-PSG.edf')]
    hypno_files = [f for f in edf_files if f.endswith('-Hypnogram.edf')]
    
    print_to_file(f"PSG files: {len(psg_files)}")
    print_to_file(f"Hypnogram files: {len(hypno_files)}")
    
    if psg_files and hypno_files:
        # Use a specific pair - ST7011J0-PSG.edf and ST7011JP-Hypnogram.edf
        psg_file = "ST7011J0-PSG.edf"
        hypno_file = "ST7011JP-Hypnogram.edf"
        
        if psg_file in psg_files and hypno_file in hypno_files:
            print_to_file(f"\nExamining specific pair: {psg_file} and {hypno_file}")
        
            # Examine the files
            raw = examine_psg_file(psg_file)
            annot = examine_hypnogram_file(hypno_file)
            
            # Save raw data info to file
            if raw is not None:
                print_to_file("\nDetailed PSG file information:")
                print_to_file(f"File: {psg_file}")
                print_to_file(f"Channels: {raw.ch_names}")
                print_to_file(f"Channel types: {raw.get_channel_types()}")
                print_to_file(f"Sampling rate: {raw.info['sfreq']} Hz")
                print_to_file(f"Duration: {raw.times[-1]:.2f} seconds ({raw.times[-1]/60:.2f} minutes)")
                
                # Get data for the first 10 seconds of the first channel
                data, times = raw[0, :int(10 * raw.info['sfreq'])]
                print_to_file(f"\nFirst 5 data points of first channel: {data[0][:5]}")
            
            # Plot a short segment of the PSG data
            if raw is not None:
                print_to_file("\nPlotting a short segment of the PSG data...")
                try:
                    # Save data to numpy file for later use
                    np.save('psg_data_sample.npy', raw.get_data())
                    print_to_file("PSG data saved to 'psg_data_sample.npy'")
                    
                    # Plot and save figure
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
                    print_to_file("Plot saved as 'psg_sample.png'")
                except Exception as e:
                    print_to_file(f"Error plotting PSG data: {str(e)}")
            
            # Create a simple hypnogram plot
            if annot is not None:
                print_to_file("\nCreating a hypnogram plot...")
                print_to_file("\nDetailed hypnogram information:")
                print_to_file(f"File: {hypno_file}")
                print_to_file(f"Number of annotations: {len(annot)}")
                
                # Print the first 10 annotations
                print_to_file("\nFirst 10 annotations:")
                for i in range(min(10, len(annot))):
                    print_to_file(f"  {i}: Onset={annot[i]['onset']:.2f}s, Duration={annot[i]['duration']:.2f}s, Description={annot[i]['description']}")
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
                
                # Plot the hypnogram
                plt.figure(figsize=(12, 6))
                plt.plot(np.arange(n_epochs) * epoch_length / 60, hypnogram, 'b-')
                plt.yticks([-1, 0, 1, 2, 3, 4], ['Unknown', 'Wake', 'N1', 'N2', 'N3', 'REM'])
                plt.title(f"Hypnogram from {hypno_file}")
                plt.xlabel('Time (minutes)')
                plt.ylabel('Sleep Stage')
                plt.grid(True)
                plt.savefig('hypnogram_sample.png')
                print_to_file("Hypnogram saved as 'hypnogram_sample.png'")
        else:
            print_to_file(f"No matching hypnogram found for {psg_file}")
    else:
        print_to_file("No PSG or hypnogram files found.")

if __name__ == "__main__":
    main()
    print_to_file("\nExamination complete.")
    output_file.close()
