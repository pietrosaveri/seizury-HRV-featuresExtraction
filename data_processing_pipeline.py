#!/usr/bin/env python3
"""
HRV Feature Processing Pipeline for Seizure Prediction Model Training

This module processes the entire OpenNeuro dataset with nested folder structure:
data/sub-XXX/ses-XX/eeg|ecg/ containing EEG/ECG files and annotations.

Uses Fixed SPH (Seizure Prediction Horizon) labeling:
- Normal: All other times (label 0)
- Pre-seizure: 180s before seizure ± 15s tolerance window (label 1) 
- Seizure: During seizure events (label 2)

Extracts HRV features from ECG and exports CSV files ready for LSTM training.
"""

import os
import glob
import pandas as pd
import numpy as np
import mne
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Import our specialized modules
from labelers import FixedSPHLabeler, WindowCenterLabeler
from ecg_processing import ECGProcessor
from hrv_features import HRVFeatureExtractor

class DataDiscovery:
    """Module for discovering and organizing dataset files."""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.subjects_data = {}
        
    def scan_dataset(self) -> Dict:
        """Scan the dataset and organize files by subject/session/run."""
        print("Scanning dataset structure...")
        
        # Find all subject directories
        subject_dirs = sorted(glob.glob(str(self.data_root / "sub-*")))
        
        for subject_dir in subject_dirs:
            subject_id = Path(subject_dir).name
            self.subjects_data[subject_id] = {}
            
            # Find all session directories for this subject
            session_dirs = sorted(glob.glob(os.path.join(subject_dir, "ses-*")))
            
            for session_dir in session_dirs:
                session_id = Path(session_dir).name
                self.subjects_data[subject_id][session_id] = {
                    'eeg_files': [],
                    'ecg_files': [],
                    'annotation_files': []
                }
                
                # Scan EEG directory
                eeg_dir = os.path.join(session_dir, "eeg")
                if os.path.exists(eeg_dir):
                    eeg_files = glob.glob(os.path.join(eeg_dir, "*_eeg.edf"))
                    tsv_files = glob.glob(os.path.join(eeg_dir, "*_events.tsv"))
                    
                    self.subjects_data[subject_id][session_id]['eeg_files'] = sorted(eeg_files)
                    self.subjects_data[subject_id][session_id]['annotation_files'] = sorted(tsv_files)
                
                # Scan ECG directory
                ecg_dir = os.path.join(session_dir, "ecg")
                if os.path.exists(ecg_dir):
                    ecg_files = glob.glob(os.path.join(ecg_dir, "*_ecg.edf"))
                    self.subjects_data[subject_id][session_id]['ecg_files'] = sorted(ecg_files)
        
        return self.subjects_data
    
    def match_runs(self) -> List[Dict]:
        """Match EEG and ECG files by run number."""
        matched_runs = []
        
        for subject_id, sessions in self.subjects_data.items():
            for session_id, files in sessions.items():
                # Extract run numbers from EEG files
                eeg_runs = {}
                for eeg_file in files['eeg_files']:
                    filename = Path(eeg_file).name
                    if '_run-' in filename:
                        run_num = filename.split('_run-')[1].split('_')[0]
                        eeg_runs[run_num] = eeg_file
                
                # Extract run numbers from ECG files
                ecg_runs = {}
                for ecg_file in files['ecg_files']:
                    filename = Path(ecg_file).name
                    if '_run-' in filename:
                        run_num = filename.split('_run-')[1].split('_')[0]
                        ecg_runs[run_num] = ecg_file
                
                # Find matching annotation files
                annotation_runs = {}
                for ann_file in files['annotation_files']:
                    filename = Path(ann_file).name
                    if '_run-' in filename:
                        run_num = filename.split('_run-')[1].split('_')[0]
                        annotation_runs[run_num] = ann_file
                
                # Match runs across modalities
                all_runs = set(eeg_runs.keys()) | set(ecg_runs.keys()) | set(annotation_runs.keys())
                
                for run_num in all_runs:
                    run_data = {
                        'subject': subject_id,
                        'session': session_id,
                        'run': run_num,
                        'eeg_file': eeg_runs.get(run_num),
                        'ecg_file': ecg_runs.get(run_num),
                        'annotation_file': annotation_runs.get(run_num)
                    }
                    matched_runs.append(run_data)
        
        return matched_runs
    
    def print_summary(self):
        """Print a summary of discovered data."""
        total_subjects = len(self.subjects_data)
        total_sessions = sum(len(sessions) for sessions in self.subjects_data.values())
        
        print(f"\nDataset Summary:")
        print(f"Total subjects: {total_subjects}")
        print(f"Total sessions: {total_sessions}")
        
        for subject_id, sessions in self.subjects_data.items():
            print(f"\n{subject_id}:")
            for session_id, files in sessions.items():
                eeg_count = len(files['eeg_files'])
                ecg_count = len(files['ecg_files'])
                ann_count = len(files['annotation_files'])
                print(f"  {session_id}: {eeg_count} EEG, {ecg_count} ECG, {ann_count} annotations")

class AnnotationProcessor:
    """Module for processing seizure annotations."""
    
    def __init__(self):
        self.event_definitions = self._load_event_definitions()
    
    def _load_event_definitions(self) -> Dict:
        """Load ILAE 2017 seizure event definitions."""
        event_definitions = {
            "1.1": "Focal Aware Motor - Automatisms",
            "1.2": "Focal Aware Motor - Atonic", 
            "1.3": "Focal Aware Motor - Clonic",
            "1.4": "Focal Aware Motor - Epileptic spasms",
            "1.5": "Focal Aware Motor - Hyperkinetic",
            "1.6": "Focal Aware Motor - Myoclonic",
            "1.7": "Focal Aware Motor - Tonic",
            "2.1": "Focal Aware Non-motor - Autonomic",
            "2.2": "Focal Aware Non-motor - Behavioral arrest",
            "2.3": "Focal Aware Non-motor - Cognitive",
            "2.4": "Focal Aware Non-motor - Emotional",
            "2.5": "Focal Aware Non-motor - Sensory",
            "3.1": "Focal Impaired Awareness Motor - Automatisms",
            "3.2": "Focal Impaired Awareness Motor - Atonic",
            "3.3": "Focal Impaired Awareness Motor - Clonic", 
            "3.4": "Focal Impaired Awareness Motor - Epileptic spasms",
            "3.5": "Focal Impaired Awareness Motor - Hyperkinetic",
            "3.6": "Focal Impaired Awareness Motor - Myoclonic",
            "3.7": "Focal Impaired Awareness Motor - Tonic",
            "4.1": "Focal Impaired Awareness Non-motor - Behavioral arrest",
            "4.2": "Focal Impaired Awareness Non-motor - Cognitive",
            "4.3": "Focal Impaired Awareness Non-motor - Emotional",
            "4.4": "Focal Impaired Awareness Non-motor - Sensory",
            "5.1": "Focal to bilateral tonic-clonic - Aware at onset",
            "5.2": "Focal to bilateral tonic-clonic - Impaired awareness at onset",
            "5.3": "Focal to bilateral tonic-clonic - Awareness unknown at onset",
            "6.1": "Generalized Motor - Tonic-clonic",
            "6.2": "Generalized Motor - Clonic", 
            "6.3": "Generalized Motor - Tonic",
            "6.4": "Generalized Motor - Myoclonic",
            "6.5": "Generalized Motor - Myoclonic-tonic-clonic",
            "6.6": "Generalized Motor - Myoclonic-atonic",
            "6.7": "Generalized Motor - Atonic",
            "6.8": "Generalized Motor - Epileptic spasms",
            "7.1": "Generalized Non-motor (absence) - Typical",
            "7.2": "Generalized Non-motor (absence) - Atypical",
            "7.3": "Generalized Non-motor (absence) - Myoclonic",
            "7.4": "Generalized Non-motor (absence) - Eyelid myoclonia"
        }
        
        return event_definitions
    
    def is_seizure_event(self, event_type: str) -> bool:
        """Check if an event type represents a seizure."""
        if pd.isna(event_type) or event_type == '':
            return False
        
        # Convert to string and check if it matches seizure patterns
        event_str = str(event_type).strip().lower()
        
        # Check if it's a numbered seizure type (1.1, 1.2, etc.)
        if event_str in self.event_definitions:
            return True
        
        # Check for seizure-specific patterns in the dataset
        # Based on observed patterns: sz_foc_*, sz_gen_*, etc.
        if event_str.startswith('sz_'):
            return True
        
        if event_type == 'sz':
            return True
            
        # Check for common seizure-related terms
        seizure_terms = [
            'seizure', 'sz', 'focal', 'generalized', 'tonic', 'clonic',
            'myoclonic', 'absence', 'atonic', 'spasm', 'automatism'
        ]
        
        return any(term in event_str for term in seizure_terms)
    
    def load_annotations(self, annotation_file: str) -> pd.DataFrame:
        """Load and process annotation file."""
        if not os.path.exists(annotation_file):
            return pd.DataFrame()
        
        try:
            annotations = pd.read_csv(annotation_file, sep='\t')
            
            # Check if we have the expected eventType column
            if 'eventType' not in annotations.columns:
                print(f"Warning: 'eventType' column not found in {annotation_file}")
                return pd.DataFrame()
            
            # Filter for seizure events
            seizure_events = annotations[
                annotations['eventType'].apply(self.is_seizure_event)
            ].copy()
            
            return seizure_events
            
        except Exception as e:
            print(f"Error loading annotations from {annotation_file}: {e}")
            return pd.DataFrame()

class HRVFeatureProcessor:
    """Module for processing ECG to HRV features with Fixed SPH labeling."""
    
    def __init__(self, sampling_rate: int = 256, sph_seconds: int = 180, 
                 label_width_seconds: int = 30, window_size_seconds: float = 30.0,
                 stride_seconds: float = 5.0):
        self.sampling_rate = sampling_rate
        self.sph_seconds = sph_seconds
        self.label_width_seconds = label_width_seconds
        self.window_size_seconds = window_size_seconds
        self.stride_seconds = stride_seconds
        
        # Initialize processing modules
        self.labeler = FixedSPHLabeler(
            sampling_rate=sampling_rate,
            sph_seconds=sph_seconds,
            label_width_seconds=label_width_seconds
        )
        self.ecg_processor = ECGProcessor(sampling_rate=sampling_rate)
        self.hrv_extractor = HRVFeatureExtractor()
        
    def process_recording(self, eeg_file: str, ecg_file: str, 
                         seizure_events: pd.DataFrame) -> pd.DataFrame:
        """Process a recording to extract HRV features with Fixed SPH labels."""
        
        # Load EEG for timing reference and create labels
        raw_eeg = mne.io.read_raw_edf(eeg_file, preload=True, verbose=False)
        raw_eeg.resample(self.sampling_rate, verbose=False)
        
        # Create Fixed SPH labels
        labels = self.labeler.create_labels(raw_eeg, seizure_events)
        
        # Load ECG data
        raw_ecg = mne.io.read_raw_edf(ecg_file, preload=True, verbose=False)
        raw_ecg.resample(self.sampling_rate, verbose=False)
        ecg_data = raw_ecg.get_data()[0]  # Assume single channel
        
        # Synchronize lengths
        min_length = min(len(ecg_data), len(labels))
        ecg_data = ecg_data[:min_length]
        labels = labels[:min_length]
        
        # Extract tachogram from ECG
        tachogram_result = self.ecg_processor.process_ecg_to_tachogram(ecg_data)
        
        if len(tachogram_result['filtered_rr']) == 0:
            return pd.DataFrame()
        
        # Extract HRV features in sliding windows
        features_df = self._extract_windowed_features(
            tachogram_result, labels, raw_eeg.times[-1]
        )
        
        # Add metadata columns
        features_df['subject_id'] = self._extract_subject_id(ecg_file)
        features_df['recording_id'] = Path(ecg_file).stem
        
        # Reorder columns
        metadata_cols = ['subject_id', 'recording_id', 'window_start_time', 
                        'window_center_time', 'window_end_time']
        feature_cols = [col for col in features_df.columns 
                       if col not in metadata_cols + ['label']]
        ordered_cols = metadata_cols + feature_cols + ['label']
        features_df = features_df[ordered_cols]
        
        return features_df
    
    def _extract_windowed_features(self, tachogram_result: Dict, 
                                 labels: np.ndarray, 
                                 total_duration: float) -> pd.DataFrame:
        """Extract HRV features in sliding windows using window center labeling."""
        rr_intervals = tachogram_result['filtered_rr']
        rr_times = tachogram_result['filtered_times']
        
        if len(rr_intervals) == 0:
            return pd.DataFrame()
        
        # Calculate number of windows
        samples_per_window = int(self.window_size_seconds * self.sampling_rate)
        step_size = int(self.stride_seconds * self.sampling_rate)
        n_samples = len(labels)
        n_windows = (n_samples - samples_per_window) // step_size + 1
        
        if n_windows <= 0:
            return pd.DataFrame()
        
        # Prepare results storage
        results = []
        feature_names = self.hrv_extractor.get_feature_names()
        
        for i in range(n_windows):
            # Calculate window boundaries
            start_sample = i * step_size
            end_sample = start_sample + samples_per_window
            center_sample = start_sample + samples_per_window // 2
            
            # Convert to time
            window_start_time = start_sample / self.sampling_rate
            window_end_time = end_sample / self.sampling_rate
            window_center_time = center_sample / self.sampling_rate
            
            # Get window label from center sample (as specified in requirements)
            window_label = labels[center_sample]
            
            # Extract RR intervals within this window
            window_rr, window_rr_times = self.ecg_processor.extract_tachogram_window(
                rr_intervals, rr_times, window_start_time, window_end_time
            )
            
            # Extract HRV features for this window
            if len(window_rr) >= 5:  # Minimum beats for meaningful features
                features = self.hrv_extractor.compute_all_features(window_rr, window_rr_times)
            else:
                # Not enough beats - use NaN features
                features = {name: np.nan for name in feature_names}
            
            # Create result row
            row = {
                'window_start_time': window_start_time,
                'window_center_time': window_center_time,
                'window_end_time': window_end_time,
                **features,
                'label': int(window_label)
            }
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def _extract_subject_id(self, filepath: str) -> str:
        """Extract subject ID from filepath."""
        filename = Path(filepath).name
        if 'sub-' in filename:
            return filename.split('sub-')[1].split('_')[0]
        return 'unknown'


class DataProcessingPipeline:
    """Main pipeline for HRV feature extraction with Fixed SPH labeling."""
    
    def __init__(self, data_root: str, output_dir: str = "hrv_features"):
        self.data_root = data_root
        self.output_dir = Path(output_dir)
        #self.output_dir.mkdir(exist_ok=True)
        
        # Initialize modules
        self.discovery = DataDiscovery(data_root)
        self.annotation_processor = AnnotationProcessor()
        self.hrv_processor = HRVFeatureProcessor()
        
        # Results storage
        self.processing_results = []
        
    def process_dataset(self):
        """Process the entire dataset to extract HRV features."""
        print("Starting HRV feature extraction pipeline...")
        print("Configuration:")
        print(f"  SPH: {self.hrv_processor.sph_seconds}s")
        print(f"  Label width: {self.hrv_processor.label_width_seconds}s") 
        print(f"  Window size: {self.hrv_processor.window_size_seconds}s")
        print(f"  Stride: {self.hrv_processor.stride_seconds}s")
        
        # Step 1: Discover data
        self.discovery.scan_dataset()
        self.discovery.print_summary()
        
        matched_runs = self.discovery.match_runs()
        print(f"\nFound {len(matched_runs)} matched runs to process")
        
        # Step 2: Process each run
        for i, run_data in enumerate(matched_runs):
            print(f"\nProcessing run {i+1}/{len(matched_runs)}: {run_data['subject']}/{run_data['session']}/run-{run_data['run']}")
            
            result = self._process_single_run(run_data)
            if result:
                self.processing_results.append(result)
        
        # Step 3: Save comprehensive results
        self._save_results()
        
    def _process_single_run(self, run_data: Dict) -> Optional[Dict]:
        """Process a single run to extract HRV features."""
        try:
            # Check if we have required files
            if not run_data['eeg_file'] or not run_data['ecg_file']:
                print(f"  Skipping - missing EEG or ECG file")
                return None
            
            # Create output filename
            output_file = self.output_dir / f"{run_data['subject']}_{run_data['session']}_run-{run_data['run']}_features.csv"
            
            # Load seizure annotations
            seizure_events = pd.DataFrame()
            if run_data['annotation_file']:
                seizure_events = self.annotation_processor.load_annotations(run_data['annotation_file'])
                print(f"  Found {len(seizure_events)} seizure events")
            
            # Process recording to extract HRV features
            print(f"  Extracting HRV features...")
            features_df = self.hrv_processor.process_recording(
                run_data['eeg_file'], run_data['ecg_file'], seizure_events
            )
            
            if features_df.empty:
                print(f"  No features extracted")
                return None
            
            # Save CSV file
            features_df.to_csv(output_file, index=False)
            
            # Calculate statistics
            label_counts = features_df['label'].value_counts().sort_index()
            total_windows = len(features_df)
            
            print(f"  Created {total_windows} windows")
            print(f"  Label distribution: {dict(label_counts)}")
            print(f"  Saved to: {output_file.name}")
            
            return {
                'subject': run_data['subject'],
                'session': run_data['session'],
                'run': run_data['run'],
                'n_windows': total_windows,
                'label_counts': label_counts.to_dict(),
                'seizure_events': len(seizure_events),
                'output_file': str(output_file)
            }
            
        except Exception as e:
            print(f"  Error processing run: {e}")
            return None
    
    def _save_results(self):
        """Save processing summary."""
        if not self.processing_results:
            print("No successful processing results to save")
            return
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(self.processing_results)
        
        # Calculate totals
        total_windows = summary_df['n_windows'].sum()
        total_seizure_events = summary_df['seizure_events'].sum()
        
        # Calculate overall label distribution
        total_normal = sum(result['label_counts'].get(0, 0) for result in self.processing_results)
        total_pre_seizure = sum(result['label_counts'].get(1, 0) for result in self.processing_results)
        total_seizure = sum(result['label_counts'].get(2, 0) for result in self.processing_results)
        
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total runs processed: {len(self.processing_results)}")
        print(f"Total windows created: {total_windows:,}")
        print(f"Total seizure events: {total_seizure_events}")
        print(f"\nOverall label distribution:")
        print(f"  Normal (0): {total_normal:,} ({total_normal/total_windows*100:.1f}%)")
        print(f"  Pre-seizure (1): {total_pre_seizure:,} ({total_pre_seizure/total_windows*100:.1f}%)")
        print(f"  Seizure (2): {total_seizure:,} ({total_seizure/total_windows*100:.1f}%)")
        
        # Save summary
        summary_file = self.output_dir / "processing_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nProcessing summary saved to: {summary_file}")
        
        # Save consolidated dataset info
        dataset_info = {
            'total_runs_processed': len(self.processing_results),
            'total_windows': int(total_windows),
            'total_seizure_events': int(total_seizure_events),
            'window_size_seconds': self.hrv_processor.window_size_seconds,
            'stride_seconds': self.hrv_processor.stride_seconds,
            'sph_seconds': self.hrv_processor.sph_seconds,
            'label_width_seconds': self.hrv_processor.label_width_seconds,
            'sampling_rate': self.hrv_processor.sampling_rate,
            'strategy': f'Fixed SPH: {self.hrv_processor.sph_seconds}s before seizure with {self.hrv_processor.label_width_seconds}s tolerance',
            'label_distribution': {
                'normal': int(total_normal),
                'pre_seizure': int(total_pre_seizure), 
                'seizure': int(total_seizure)
            }
        }

        with open(self.output_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
            
        print(f"Dataset info saved to: {self.output_dir / 'dataset_info.json'}")
        print(f"CSV feature files saved in: {self.output_dir}")
        
        # Create combined CSV
        print(f"\nCreating combined features CSV...")
        all_csvs = list(self.output_dir.glob("*_features.csv"))
        if all_csvs:
            combined_dfs = []
            for csv_file in all_csvs:
                df = pd.read_csv(csv_file)
                combined_dfs.append(df)
            
            combined_df = pd.concat(combined_dfs, ignore_index=True)
            combined_file = self.output_dir / "combined_features.csv"
            combined_df.to_csv(combined_file, index=False)
            print(f"Combined features saved to: {combined_file}")
            print(f"Total combined windows: {len(combined_df):,}")

def main():
    """Main function to run the HRV feature extraction pipeline."""
    # Configuration
    #data_root = "/Volumes/Seizury/ds005873"
    #output_dir = "/Volumes/Seizury/HRV/hrv_features"

    data_root = "s3://seizury-data/ds005873"
    output_dir = "s3://seizury-data/hrv_features"

    # Create and run pipeline
    pipeline = DataProcessingPipeline(data_root, output_dir)
    pipeline.process_dataset()

if __name__ == "__main__":
    main()