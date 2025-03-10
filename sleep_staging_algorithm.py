#!/usr/bin/env python
# coding: utf-8

"""
Sleep Staging Algorithm

This script implements a machine learning-based sleep staging algorithm that:
1. Processes EEG data from .edf files
2. Extracts relevant features
3. Trains a classifier to predict sleep stages
4. Evaluates the model performance
5. Visualizes the results
6. Allows for uploading new annotated files
7. Can score unannotated files

Sleep stages are classified according to the AASM (American Academy of Sleep Medicine) standards:
- Wake (W)
- N1 (light sleep)
- N2 (intermediate sleep)
- N3 (deep sleep, combining stages 3 and 4 from R&K)
- REM (rapid eye movement)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC
import mne
from mne.time_frequency import psd_welch
import glob
from tqdm import tqdm
import joblib
import datetime

# Set random seed for reproducibility
np.random.seed(42)

class SleepStagingAlgorithm:
    def __init__(self, data_dir=None, model_dir='models'):
        """
        Initialize the sleep staging algorithm.
        
        Parameters:
        -----------
        data_dir : str, optional
            Directory containing the .edf files
        model_dir : str
            Directory to save/load trained models
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.psg_files = []
        self.hypno_files = []
        self.X = None  # Features
        self.y = None  # Labels (sleep stages)
        self.model = None
        self.scaler = StandardScaler()
        self.epoch_duration = 30  # Standard 30-second epochs for sleep staging
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def load_dataset(self, data_dir=None):
        """
        Load all PSG and hypnogram files from the data directory.
        
        Parameters:
        -----------
        data_dir : str, optional
            Directory containing the .edf files. If None, use self.data_dir.
        
        Returns:
        --------
        self
        """
        if data_dir is not None:
            self.data_dir = data_dir
        
        if self.data_dir is None:
            raise ValueError("No data directory specified.")
        
        print(f"Loading dataset from {self.data_dir}...")
        
        # Find all PSG and hypnogram files
        self.psg_files = sorted(glob.glob(os.path.join(self.data_dir, "*-PSG.edf")))
        self.hypno_files = sorted(glob.glob(os.path.join(self.data_dir, "*-Hypnogram.edf")))
        
        if not self.psg_files:
            raise ValueError(f"No PSG files found in {self.data_dir}")
        
        if not self.hypno_files:
            raise ValueError(f"No hypnogram files found in {self.data_dir}")
        
        print(f"Found {len(self.psg_files)} PSG files and {len(self.hypno_files)} hypnogram files.")
        
        # Verify that each PSG file has a corresponding hypnogram file
        psg_base_names = [os.path.basename(f).split('-PSG')[0] for f in self.psg_files]
        hypno_base_names = [os.path.basename(f).split('-Hypnogram')[0] for f in self.hypno_files]
        
        # Find matching pairs
        self.matched_files = []
        for i, psg_base in enumerate(psg_base_names):
            matching_hypnos = [j for j, hypno_base in enumerate(hypno_base_names) if hypno_base.startswith(psg_base)]
            if matching_hypnos:
                self.matched_files.append((self.psg_files[i], self.hypno_files[matching_hypnos[0]]))
        
        print(f"Found {len(self.matched_files)} matched PSG-hypnogram pairs.")
        
        return self
    
    def preprocess_file(self, psg_file):
        """
        Preprocess a single PSG file.
        
        Parameters:
        -----------
        psg_file : str
            Path to the PSG file
            
        Returns:
        --------
        raw : mne.io.Raw
            Preprocessed MNE Raw object
        """
        print(f"Preprocessing {os.path.basename(psg_file)}...")
        
        # Read the PSG file
        raw = mne.io.read_raw_edf(psg_file, preload=True)
        
        # Apply filters
        raw.filter(l_freq=0.5, h_freq=45)  # Bandpass filter
        raw.notch_filter(freqs=np.arange(50, 251, 50))  # Notch filter for line noise
        
        return raw
    
    def extract_features(self, raw, epoch_duration=30):
        """
        Extract features from preprocessed EEG data.
        
        Parameters:
        -----------
        raw : mne.io.Raw
            Preprocessed MNE Raw object
        epoch_duration : int
            Duration of each epoch in seconds
            
        Returns:
        --------
        features : numpy.ndarray
            Extracted features for each epoch
        """
        # Calculate the number of samples per epoch
        samples_per_epoch = int(epoch_duration * raw.info['sfreq'])
        
        # Calculate the number of epochs
        n_epochs = int(raw.n_times / samples_per_epoch)
        
        # Initialize feature array
        n_channels = len(raw.ch_names)
        features = np.zeros((n_epochs, n_channels * 5))  # 5 features per channel
        
        # Frequency bands (Hz)
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        # Extract features for each epoch
        for i in range(n_epochs):
            start_sample = i * samples_per_epoch
            end_sample = start_sample + samples_per_epoch
            
            # Extract epoch data
            epoch_data = raw.get_data(start=start_sample, stop=end_sample)
            
            # Calculate power spectral density
            psds, freqs = psd_welch(raw, fmin=0.5, fmax=45, tmin=start_sample/raw.info['sfreq'], 
                                   tmax=end_sample/raw.info['sfreq'], n_fft=int(raw.info['sfreq'] * 2))
            
            # Calculate band powers for each channel
            for j, ch_name in enumerate(raw.ch_names):
                for k, (band_name, (fmin, fmax)) in enumerate(bands.items()):
                    # Find frequencies within the band
                    freq_mask = (freqs >= fmin) & (freqs <= fmax)
                    # Calculate band power
                    band_power = np.mean(psds[j, :, freq_mask], axis=1)
                    # Store feature
                    features[i, j * 5 + k] = band_power
        
        return features
    
    def extract_labels(self, hypno_file, n_epochs):
        """
        Extract sleep stage labels from a hypnogram file.
        
        Parameters:
        -----------
        hypno_file : str
            Path to the hypnogram file
        n_epochs : int
            Number of epochs to extract labels for
            
        Returns:
        --------
        labels : numpy.ndarray
            Sleep stage labels for each epoch
        """
        # Read the hypnogram file
        annot = mne.read_annotations(hypno_file)
        
        # Initialize labels array
        labels = np.zeros(n_epochs) - 1  # -1 for unknown
        
        # Stage mapping
        stage_map = {
            'Sleep stage W': 0,      # Wake
            'Sleep stage 1': 1,      # N1
            'Sleep stage 2': 2,      # N2
            'Sleep stage 3': 3,      # N3
            'Sleep stage 4': 3,      # N3 (combining stages 3 and 4)
            'Sleep stage R': 4,      # REM
            'Movement time': -1,     # Unknown
            'Sleep stage ?': -1      # Unknown
        }
        
        # Fill in the labels
        for a in annot:
            onset_epoch = int(a['onset'] / self.epoch_duration)
            duration_epochs = int(a['duration'] / self.epoch_duration)
            if a['description'] in stage_map:
                stage = stage_map[a['description']]
                for i in range(duration_epochs):
                    if onset_epoch + i < n_epochs:
                        labels[onset_epoch + i] = stage
        
        return labels
    
    def process_dataset(self):
        """
        Process the entire dataset, extracting features and labels.
        
        Returns:
        --------
        self
        """
        if not hasattr(self, 'matched_files') or not self.matched_files:
            raise ValueError("No matched files found. Run load_dataset first.")
        
        print("Processing dataset...")
        
        all_features = []
        all_labels = []
        
        for psg_file, hypno_file in tqdm(self.matched_files):
            # Preprocess the PSG file
            raw = self.preprocess_file(psg_file)
            
            # Extract features
            features = self.extract_features(raw, self.epoch_duration)
            
            # Extract labels
            labels = self.extract_labels(hypno_file, features.shape[0])
            
            # Store features and labels
            all_features.append(features)
            all_labels.append(labels)
        
        # Combine features and labels from all files
        self.X = np.vstack(all_features)
        self.y = np.concatenate(all_labels)
        
        # Remove epochs with unknown labels
        valid_idx = self.y >= 0
        self.X = self.X[valid_idx]
        self.y = self.y[valid_idx]
        
        print(f"Processed {len(self.matched_files)} files.")
        print(f"Extracted {self.X.shape[0]} epochs with valid labels.")
        print(f"Feature shape: {self.X.shape}")
        
        # Print class distribution
        unique_labels, counts = np.unique(self.y, return_counts=True)
        print("Class distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"  {self.label_to_stage(label)}: {count} epochs ({count * self.epoch_duration / 60:.1f} minutes)")
        
        return self
    
    def train_model(self, model_type='random_forest', save_model=True):
        """
        Train a machine learning model for sleep staging.
        
        Parameters:
        -----------
        model_type : str
            Type of model to train ('random_forest' or 'svm')
        save_model : bool
            Whether to save the trained model
            
        Returns:
        --------
        self
        """
        if self.X is None or self.y is None:
            raise ValueError("Features and/or labels not available. Run process_dataset first.")
        
        print(f"Training {model_type} model...")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train the model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'svm':
            self.model = SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(
            y_test, 
            y_pred,
            target_names=[self.label_to_stage(i) for i in range(5)]
        ))
        
        # Store test data for later visualization
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        
        # Save the model if requested
        if save_model:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = os.path.join(self.model_dir, f"sleep_staging_{model_type}_{timestamp}.joblib")
            scaler_filename = os.path.join(self.model_dir, f"scaler_{model_type}_{timestamp}.joblib")
            
            joblib.dump(self.model, model_filename)
            joblib.dump(self.scaler, scaler_filename)
            
            print(f"Model saved to {model_filename}")
            print(f"Scaler saved to {scaler_filename}")
        
        return self
    
    def load_model(self, model_path, scaler_path=None):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
        scaler_path : str, optional
            Path to the saved scaler. If None, will try to infer from model_path.
            
        Returns:
        --------
        self
        """
        print(f"Loading model from {model_path}...")
        
        # Load the model
        self.model = joblib.load(model_path)
        
        # Load the scaler
        if scaler_path is None:
            # Try to infer scaler path from model path
            scaler_path = model_path.replace("sleep_staging_", "scaler_")
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from {scaler_path}")
        else:
            print(f"Warning: Scaler not found at {scaler_path}. Using default scaler.")
            self.scaler = StandardScaler()
        
        return self
    
    def visualize_results(self, save_dir=None):
        """
        Visualize the results of the sleep staging algorithm.
        
        Parameters:
        -----------
        save_dir : str, optional
            Directory to save the visualizations. If None, will use current directory.
            
        Returns:
        --------
        self
        """
        if not hasattr(self, 'y_pred'):
            raise ValueError("Model predictions not available. Run train_model first.")
        
        print("Visualizing results...")
        
        # Create save directory if specified
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Confusion Matrix
        plt.subplot(2, 2, 1)
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[self.label_to_stage(i) for i in range(5)],
            yticklabels=[self.label_to_stage(i) for i in range(5)]
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # 2. Feature Importance (for Random Forest)
        if hasattr(self.model, 'feature_importances_'):
            plt.subplot(2, 2, 2)
            # Create feature names
            feature_names = []
            for ch_idx in range(self.X.shape[1] // 5):
                for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                    feature_names.append(f"Ch{ch_idx}_{band}")
            
            # Sort features by importance
            feature_importance = self.model.feature_importances_
            sorted_idx = np.argsort(feature_importance)[-20:]  # Top 20 features
            
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
            plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
            plt.title('Top 20 Feature Importance')
            plt.xlabel('Importance')
        
        # 3. Hypnogram (sleep stage over time)
        plt.subplot(2, 1, 2)
        # Get a continuous segment of predictions
        segment_length = min(500, len(self.y_pred))
        time = np.arange(segment_length) * self.epoch_duration / 60  # Convert to minutes
        plt.plot(time, self.y_test[:segment_length], 'b-', label='True')
        plt.plot(time, self.y_pred[:segment_length], 'r--', label='Predicted')
        plt.yticks(range(5), [self.label_to_stage(i) for i in range(5)])
        plt.title('Hypnogram (Sleep Stages Over Time)')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Sleep Stage')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the figure
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, 'sleep_staging_results.png'))
        else:
            plt.savefig('sleep_staging_results.png')
        
        plt.show()
        
        return self
    
    def upload_annotated_file(self, psg_file, hypno_file, process=True):
        """
        Upload a new annotated file pair and optionally process it.
        
        Parameters:
        -----------
        psg_file : str
            Path to the PSG file
        hypno_file : str
            Path to the hypnogram file
        process : bool
            Whether to process the file and add it to the dataset
            
        Returns:
        --------
        features, labels : tuple
            Extracted features and labels if process=True, otherwise None
        """
        print(f"Uploading annotated file pair: {os.path.basename(psg_file)} and {os.path.basename(hypno_file)}")
        
        # Check if files exist
        if not os.path.exists(psg_file):
            raise ValueError(f"PSG file not found: {psg_file}")
        
        if not os.path.exists(hypno_file):
            raise ValueError(f"Hypnogram file not found: {hypno_file}")
        
        # Add to matched files
        self.matched_files.append((psg_file, hypno_file))
        
        if process:
            # Preprocess the PSG file
            raw = self.preprocess_file(psg_file)
            
            # Extract features
            features = self.extract_features(raw, self.epoch_duration)
            
            # Extract labels
            labels = self.extract_labels(hypno_file, features.shape[0])
            
            # Add to dataset if it exists
            if self.X is not None and self.y is not None:
                # Remove epochs with unknown labels
                valid_idx = labels >= 0
                valid_features = features[valid_idx]
                valid_labels = labels[valid_idx]
                
                # Add to dataset
                self.X = np.vstack([self.X, valid_features])
                self.y = np.concatenate([self.y, valid_labels])
                
                print(f"Added {valid_features.shape[0]} epochs with valid labels to the dataset.")
                print(f"New dataset size: {self.X.shape[0]} epochs")
            
            return features, labels
        
        return None, None
    
    def score_file(self, psg_file, save_dir=None):
        """
        Score an unannotated PSG file.
        
        Parameters:
        -----------
        psg_file : str
            Path to the PSG file
        save_dir : str, optional
            Directory to save the results. If None, will use current directory.
            
        Returns:
        --------
        predictions, hypnogram : tuple
            Predicted sleep stages and hypnogram
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Run train_model or load_model first.")
        
        print(f"Scoring file: {os.path.basename(psg_file)}")
        
        # Check if file exists
        if not os.path.exists(psg_file):
            raise ValueError(f"PSG file not found: {psg_file}")
        
        # Create save directory if specified
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Preprocess the PSG file
        raw = self.preprocess_file(psg_file)
        
        # Extract features
        features = self.extract_features(raw, self.epoch_duration)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        
        # Convert numeric predictions to stage labels
        stage_predictions = [self.label_to_stage(pred) for pred in predictions]
        
        # Create a hypnogram
        plt.figure(figsize=(12, 6))
        time = np.arange(len(predictions)) * self.epoch_duration / 60  # Convert to minutes
        plt.plot(time, predictions, 'b-')
        plt.yticks(range(5), [self.label_to_stage(i) for i in range(5)])
        plt.title(f'Predicted Hypnogram for {os.path.basename(psg_file)}')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Sleep Stage')
        plt.grid(True)
        
        # Save the hypnogram
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f'hypnogram_{os.path.basename(psg_file)}.png'))
        else:
            plt.savefig(f'hypnogram_{os.path.basename(psg_file)}.png')
        
        plt.show()
        
        # Save predictions to CSV
        results_df = pd.DataFrame({
            'Epoch': range(len(predictions)),
            'Time (min)': time,
            'Prediction': predictions,
            'Sleep Stage': stage_predictions
        })
        
        if save_dir is not None:
            results_df.to_csv(os.path.join(save_dir, f'predictions_{os.path.basename(psg_file)}.csv'), index=False)
        else:
            results_df.to_csv(f'predictions_{os.path.basename(psg_file)}.csv', index=False)
        
        print(f"Predictions saved to CSV.")
        
        # Print summary
        print("\nSleep Stage Summary:")
        for stage in range(5):
            count = np.sum(predictions == stage)
            print(f"  {self.label_to_stage(stage)}: {count} epochs ({count * self.epoch_duration / 60:.1f} minutes)")
        
        return predictions, stage_predictions
    
    def label_to_stage(self, label):
        """
        Convert numeric label to sleep stage string.
        
        Parameters:
        -----------
        label : int
            Numeric label for the sleep stage
            
        Returns:
        --------
        stage : str
            Sleep stage string
        """
        label_map = {
            0: 'Wake',
            1: 'N1',
            2: 'N2',
            3: 'N3',
            4: 'REM',
            -1: 'Unknown'
        }
        
        return label_map.get(label, 'Unknown')
    
    def stage_to_label(self, stage):
        """
        Convert sleep stage string to numeric label.
        
        Parameters:
        -----------
        stage : str
            Sleep stage string
            
        Returns:
        --------
        label : int
            Numeric label for the sleep stage
        """
        stage_map = {
            'Wake': 0,
            'N1': 1,
            'N2': 2,
            'N3': 3,
            'REM': 4,
            'Unknown': -1
        }
        
        return stage_map.get(stage, -1)


# Example usage
if __name__ == "__main__":
    # Directory containing .edf files
    data_dir = "."  # Current directory
    
    # Create and run the sleep staging algorithm
    sleep_stager = SleepStagingAlgorithm(data_dir)
    
    # Load and process the dataset
    sleep_stager.load_dataset()
    sleep_stager.process_dataset()
    
    # Train the model
    sleep_stager.train_model(model_type='random_forest')
    
    # Visualize results
    sleep_stager.visualize_results()
    
    # Score a new file
    # sleep_stager.score_file("path/to/new/file.edf")
