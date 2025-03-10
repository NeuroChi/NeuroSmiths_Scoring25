#!/usr/bin/env python
# coding: utf-8

"""
Sleep Staging Demo

This script demonstrates how to use the SleepStagingAlgorithm class to:
1. Train a model on the dataset
2. Upload new annotated files
3. Score unannotated files

Usage:
    python sleep_staging_demo.py [--train] [--score FILE] [--upload PSG_FILE HYPNO_FILE]

Options:
    --train             Train a new model on the dataset
    --score FILE        Score an unannotated PSG file
    --upload PSG HYPNO  Upload a new annotated PSG and hypnogram file pair
    --model MODEL       Path to a trained model to load (optional)
    --data_dir DIR      Path to the data directory (default: current directory)
    --output_dir DIR    Path to the output directory (default: 'results')
"""

import os
import sys
import argparse
import glob
from sleep_staging_algorithm import SleepStagingAlgorithm

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Sleep Staging Demo')
    parser.add_argument('--train', action='store_true', help='Train a new model on the dataset')
    parser.add_argument('--score', type=str, help='Score an unannotated PSG file')
    parser.add_argument('--upload', nargs=2, metavar=('PSG_FILE', 'HYPNO_FILE'), help='Upload a new annotated PSG and hypnogram file pair')
    parser.add_argument('--model', type=str, help='Path to a trained model to load')
    parser.add_argument('--data_dir', type=str, default='.', help='Path to the data directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Path to the output directory')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create the sleep staging algorithm
    sleep_stager = SleepStagingAlgorithm(data_dir=args.data_dir, model_dir=args.output_dir)
    
    # Load a trained model if specified
    if args.model:
        if os.path.exists(args.model):
            sleep_stager.load_model(args.model)
        else:
            print(f"Error: Model file not found: {args.model}")
            return
    
    # Train a new model
    if args.train:
        print("\n=== Training a new model ===")
        
        # Load and process the dataset
        sleep_stager.load_dataset()
        sleep_stager.process_dataset()
        
        # Ask for model type
        model_type = input("Select model type (random_forest/svm) [default: random_forest]: ").strip()
        if not model_type:
            model_type = 'random_forest'
        
        # Train the model
        sleep_stager.train_model(model_type=model_type)
        
        # Visualize results
        sleep_stager.visualize_results(save_dir=args.output_dir)
    
    # Upload a new annotated file pair
    if args.upload:
        psg_file, hypno_file = args.upload
        
        print(f"\n=== Uploading annotated file pair ===")
        print(f"PSG file: {psg_file}")
        print(f"Hypnogram file: {hypno_file}")
        
        try:
            # Check if files exist
            if not os.path.exists(psg_file):
                print(f"Error: PSG file not found: {psg_file}")
                return
            
            if not os.path.exists(hypno_file):
                print(f"Error: Hypnogram file not found: {hypno_file}")
                return
            
            # Upload the file pair
            features, labels = sleep_stager.upload_annotated_file(psg_file, hypno_file)
            
            # Ask if user wants to retrain the model
            retrain = input("Do you want to retrain the model with the new data? (y/n) [default: n]: ").strip().lower()
            if retrain == 'y':
                # Train the model
                model_type = input("Select model type (random_forest/svm) [default: random_forest]: ").strip()
                if not model_type:
                    model_type = 'random_forest'
                
                sleep_stager.train_model(model_type=model_type)
                
                # Visualize results
                sleep_stager.visualize_results(save_dir=args.output_dir)
        
        except Exception as e:
            print(f"Error uploading file pair: {str(e)}")
    
    # Score an unannotated file
    if args.score:
        print(f"\n=== Scoring unannotated file ===")
        print(f"PSG file: {args.score}")
        
        try:
            # Check if file exists
            if not os.path.exists(args.score):
                print(f"Error: PSG file not found: {args.score}")
                return
            
            # Check if a model is loaded
            if sleep_stager.model is None:
                # Try to find a model in the output directory
                model_files = glob.glob(os.path.join(args.output_dir, "sleep_staging_*.joblib"))
                if model_files:
                    # Use the most recent model
                    model_file = sorted(model_files)[-1]
                    print(f"Loading model from {model_file}")
                    sleep_stager.load_model(model_file)
                else:
                    print("Error: No model loaded or found. Please train a model first or specify a model with --model.")
                    return
            
            # Score the file
            predictions, stage_predictions = sleep_stager.score_file(args.score, save_dir=args.output_dir)
            
            print(f"\nResults saved to {args.output_dir}")
        
        except Exception as e:
            print(f"Error scoring file: {str(e)}")
    
    # If no action specified, print help
    if not (args.train or args.upload or args.score):
        print(__doc__)

if __name__ == "__main__":
    main()
