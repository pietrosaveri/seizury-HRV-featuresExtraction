#!/usr/bin/env python3
"""
HRV Feature Pipeline for Seizure Prediction

This module implements the complete pipeline for extracting HRV features
from ECG data with Fixed SPH labeling for seizure prediction at exactly 
180 seconds before seizure onset.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import mne
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Import our custom modules
from labelers import FixedSPHLabeler, WindowCenterLabeler
from ecg_processing import ECGProcessor
from hrv_features import HRVFeatureExtractor
from data_processing_pipeline import AnnotationProcessor


class HRVFeaturePipeline:
    """
    Complete pipeline for HRV feature extraction with Fixed SPH labeling.
    """
    
    def __init__(self, 
                 sampling_rate: int = 256,
                 sph_seconds: int = 180,
                 label_width_seconds: int = 30,
                 window_size_seconds: float = 30.0,
                 stride_seconds: float = 5.0,
                 output_dir: str = "hrv_features",
                 resume: bool = False):
        """
        Initialize HRV feature pipeline.
        
        Args:
            sampling_rate: Target sampling rate for processing
            sph_seconds: Seizure prediction horizon (default: 180s)
            label_width_seconds: Width of positive label window (default: 30s)
            window_size_seconds: Window size for feature extraction (default: 30s)
            stride_seconds: Stride between windows (default: 5s)
            output_dir: Output directory for CSV files
            resume: If True, skip already processed files (default: False)
        """
        self.sampling_rate = sampling_rate
        self.sph_seconds = sph_seconds
        self.label_width_seconds = label_width_seconds
        self.window_size_seconds = window_size_seconds
        self.stride_seconds = stride_seconds
        self.resume = resume
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize processing modules
        self.labeler = FixedSPHLabeler(
            sampling_rate=sampling_rate,
            sph_seconds=sph_seconds,
            label_width_seconds=label_width_seconds
        )
        
        self.ecg_processor = ECGProcessor(sampling_rate=sampling_rate)
        self.hrv_extractor = HRVFeatureExtractor()
        self.annotation_processor = AnnotationProcessor()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def process_recording(self, eeg_file: str, ecg_file: str, 
                         annotation_file: Optional[str] = None) -> pd.DataFrame:
        """
        Process a single recording to extract HRV features with labels.
        
        Args:
            eeg_file: Path to EEG file (for timing reference)
            ecg_file: Path to ECG file
            annotation_file: Path to seizure annotation file
            
        Returns:
            DataFrame with HRV features and labels per window
        """
        self.logger.info(f"Processing recording: {Path(ecg_file).name}")
        
        try:
            # Load EEG for timing reference
            self.logger.info("Loading EEG data for timing reference...")
            raw_eeg = mne.io.read_raw_edf(eeg_file, preload=True, verbose=False)
            raw_eeg.resample(self.sampling_rate, verbose=False)
            
            # Load seizure annotations
            seizure_events = pd.DataFrame()
            if annotation_file and os.path.exists(annotation_file):
                seizure_events = self.annotation_processor.load_annotations(annotation_file)
                self.logger.info(f"Found {len(seizure_events)} seizure events")
            else:
                self.logger.warning("No seizure annotations found")
            
            # Create SPH labels
            self.logger.info("Creating Fixed SPH labels...")
            labels = self.labeler.create_labels(raw_eeg, seizure_events)
            
            # Load ECG data
            self.logger.info("Loading ECG data...")
            raw_ecg = mne.io.read_raw_edf(ecg_file, preload=True, verbose=False)
            raw_ecg.resample(self.sampling_rate, verbose=False)
            ecg_data = raw_ecg.get_data()[0]  # Assume single channel
            
            # Synchronize lengths
            min_length = min(len(ecg_data), len(labels))
            ecg_data = ecg_data[:min_length]
            labels = labels[:min_length]
            
            # Extract tachogram from ECG
            self.logger.info("Extracting R-peaks and computing RR intervals...")
            tachogram_result = self.ecg_processor.process_ecg_to_tachogram(ecg_data)
            
            if len(tachogram_result['filtered_rr']) == 0:
                self.logger.warning("No valid RR intervals found")
                return pd.DataFrame()
            
            # Extract HRV features in sliding windows
            self.logger.info("Extracting HRV features in sliding windows...")
            features_df = self._extract_windowed_features(
                tachogram_result, labels, raw_eeg.times[-1]
            )
            
            # Add metadata columns
            features_df['subject_id'] = self._extract_subject_id(ecg_file)
            features_df['recording_id'] = Path(ecg_file).stem
            
            # Reorder columns to put metadata first
            metadata_cols = ['subject_id', 'recording_id', 'window_start_time', 
                           'window_center_time', 'window_end_time']
            feature_cols = [col for col in features_df.columns 
                          if col not in metadata_cols + ['label']]
            ordered_cols = metadata_cols + feature_cols + ['label']
            features_df = features_df[ordered_cols]
            
            self.logger.info(f"Extracted {len(features_df)} windows with features")
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error processing recording: {e}")
            return pd.DataFrame()
    
    def _extract_windowed_features(self, tachogram_result: Dict, 
                                 labels: np.ndarray, 
                                 total_duration: float) -> pd.DataFrame:
        """
        Extract HRV features in sliding windows.
        
        Args:
            tachogram_result: Result from ECG processing
            labels: Label array for the entire recording
            total_duration: Total duration of recording in seconds
            
        Returns:
            DataFrame with features per window
        """
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
            
            # Get window label from center sample
            window_label = labels[center_sample]
            
            # Extract RR intervals within this window
            window_rr, window_rr_times = self.ecg_processor.extract_tachogram_window(
                rr_intervals, rr_times, window_start_time, window_end_time
            )
            
            # Extract HRV features for this window
            if len(window_rr) >= 5:  # Minimum beats for meaningful features
                features = self.hrv_extractor.compute_all_features(window_rr, window_rr_times)
            else:
                # Not enough beats - use empty features
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
    
    def save_features_csv(self, features_df: pd.DataFrame, output_path: str):
        """Save features DataFrame to CSV."""
        features_df.to_csv(output_path, index=False)
        self.logger.info(f"Saved features to: {output_path}")
        
        # Log statistics
        if not features_df.empty:
            label_counts = features_df['label'].value_counts().sort_index()
            self.logger.info(f"Label distribution: {dict(label_counts)}")
            
            # Check for NaN features
            nan_features = features_df.isnull().sum()
            if nan_features.any():
                self.logger.warning(f"Features with NaN values: {nan_features[nan_features > 0].to_dict()}")
    
    def _get_output_file_path(self, recording_id: str) -> Path:
        """Get the output file path for a recording."""
        return self.output_dir / f"{recording_id}_features.csv"
    
    def _is_file_already_processed(self, recording_id: str, source_files: List[str]) -> bool:
        """
        Check if a recording has already been processed and is up-to-date.
        
        Args:
            recording_id: ID of the recording
            source_files: List of source files (EEG, ECG, annotation)
            
        Returns:
            True if output file exists and is newer than all source files
        """
        output_file = self._get_output_file_path(recording_id)
        
        if not output_file.exists():
            return False
        
        # Check if output file is newer than all source files
        output_mtime = output_file.stat().st_mtime
        
        for source_file in source_files:
            if source_file and Path(source_file).exists():
                source_mtime = Path(source_file).stat().st_mtime
                if source_mtime > output_mtime:
                    self.logger.info(f"Source file {source_file} is newer than output file")
                    return False
        
        # Check if output file has expected structure
        try:
            df = pd.read_csv(output_file)
            
            # Basic validation: check for required columns
            required_cols = ['window_center_time', 'label', 'RMSSD', 'SDNN']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.logger.warning(f"Output file {output_file} missing columns: {missing_cols}")
                return False
            
            # Check if file is empty
            if len(df) == 0:
                self.logger.warning(f"Output file {output_file} is empty")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Cannot validate output file {output_file}: {e}")
            return False
    
    def _get_config_hash(self) -> str:
        """
        Generate a hash of current configuration to detect parameter changes.
        """
        import hashlib
        config_str = f"{self.sph_seconds}_{self.label_width_seconds}_{self.window_size_seconds}_{self.stride_seconds}_{self.sampling_rate}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _save_processing_metadata(self, recording_id: str, config_hash: str):
        """Save metadata about processing for future resume checks."""
        metadata_file = self.output_dir / f"{recording_id}_metadata.json"
        metadata = {
            'recording_id': recording_id,
            'config_hash': config_hash,
            'processing_time': pd.Timestamp.now().isoformat(),
            'pipeline_version': '1.1',
            'sph_seconds': self.sph_seconds,
            'label_width_seconds': self.label_width_seconds,
            'window_size_seconds': self.window_size_seconds,
            'stride_seconds': self.stride_seconds,
            'sampling_rate': self.sampling_rate
        }
        
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _check_config_compatibility(self, recording_id: str) -> bool:
        """Check if existing output was generated with compatible config."""
        metadata_file = self.output_dir / f"{recording_id}_metadata.json"
        
        if not metadata_file.exists():
            return False
        
        try:
            import json
            with open(metadata_file, 'r') as f:
                saved_metadata = json.load(f)
            
            current_hash = self._get_config_hash()
            saved_hash = saved_metadata.get('config_hash', '')
            
            if current_hash != saved_hash:
                self.logger.info(f"Config changed for {recording_id}. Current: {current_hash}, Saved: {saved_hash}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Cannot read metadata for {recording_id}: {e}")
            return False
    
    def process_dataset(self, dataset_config: Dict) -> pd.DataFrame:
        """
        Process entire dataset according to configuration.
        
        Args:
            dataset_config: Dictionary with dataset configuration
            
        Returns:
            Combined DataFrame with all recordings
        """
        all_features = []
        skipped_count = 0
        processed_count = 0
        total_count = len(dataset_config['recordings'])
        
        self.logger.info(f"Found {total_count} recordings to process")
        
        if self.resume:
            self.logger.info("Resume mode enabled - will skip already processed files")
        
        for i, recording in enumerate(dataset_config['recordings'], 1):
            recording_id = recording['id']
            self.logger.info(f"[{i}/{total_count}] Checking {recording_id}...")
            
            # Check if already processed (resume functionality)
            if self.resume:
                source_files = [
                    recording['eeg_file'],
                    recording['ecg_file'],
                    recording.get('annotation_file')
                ]
                
                # Check if file exists, is up-to-date, and config is compatible
                if (self._is_file_already_processed(recording_id, source_files) and 
                    self._check_config_compatibility(recording_id)):
                    
                    self.logger.info(f"Skipping {recording_id} (already processed)")
                    
                    # Load existing features to include in combined output
                    try:
                        output_file = self._get_output_file_path(recording_id)
                        existing_df = pd.read_csv(output_file)
                        all_features.append(existing_df)
                        skipped_count += 1
                        continue
                    except Exception as e:
                        self.logger.warning(f"Failed to load existing file for {recording_id}: {e}")
                        # Fall through to reprocess
            
            # Process the recording
            self.logger.info(f"Processing {recording_id}...")
            
            features_df = self.process_recording(
                eeg_file=recording['eeg_file'],
                ecg_file=recording['ecg_file'],
                annotation_file=recording.get('annotation_file')
            )
            
            if not features_df.empty:
                # Save individual recording CSV
                output_file = self._get_output_file_path(recording_id)
                self.save_features_csv(features_df, output_file)
                
                # Save processing metadata for future resume checks
                config_hash = self._get_config_hash()
                self._save_processing_metadata(recording_id, config_hash)
                
                all_features.append(features_df)
                processed_count += 1
                self.logger.info(f"Completed {recording_id} ({len(features_df)} windows)")
            else:
                self.logger.warning(f"No features extracted for {recording_id}")
        
        # Summary
        self.logger.info(f"\nProcessing Summary:")
        self.logger.info(f"   Total recordings: {total_count}")
        self.logger.info(f"   Processed: {processed_count}")
        self.logger.info(f"   Skipped (already done): {skipped_count}")
        self.logger.info(f"   Failed: {total_count - processed_count - skipped_count}")
        
        # Combine all recordings
        if all_features:
            combined_df = pd.concat(all_features, ignore_index=True)
            
            # Save combined dataset
            combined_file = self.output_dir / "combined_features.csv"
            self.save_features_csv(combined_df, combined_file)
            
            self.logger.info(f"Saved combined features: {len(combined_df)} total windows")
            
            return combined_df
        
        return pd.DataFrame()


def create_dataset_config(data_root: str) -> Dict:
    """
    Create dataset configuration by scanning data directory.
    
    Args:
        data_root: Root directory of dataset
        
    Returns:
        Dataset configuration dictionary
    """
    from data_processing_pipeline import DataDiscovery
    
    discovery = DataDiscovery(data_root)
    discovery.scan_dataset()
    matched_runs = discovery.match_runs()
    
    recordings = []
    for run_data in matched_runs:
        if run_data['eeg_file'] and run_data['ecg_file']:
            recording_id = f"{run_data['subject']}_{run_data['session']}_run-{run_data['run']}"
            recordings.append({
                'id': recording_id,
                'subject': run_data['subject'],
                'session': run_data['session'],
                'run': run_data['run'],
                'eeg_file': run_data['eeg_file'],
                'ecg_file': run_data['ecg_file'],
                'annotation_file': run_data['annotation_file']
            })
    
    return {'recordings': recordings}


def main():
    """Main entry point for HRV feature extraction pipeline."""
    parser = argparse.ArgumentParser(description='Extract HRV features for seizure prediction')
    parser.add_argument('--data-root', required=True, help='Root directory of dataset')
    parser.add_argument('--output-dir', default='/Volumes/Seizury/HRV/hrv_features', help='Output directory')
    parser.add_argument('--sph', type=int, default=180, help='Seizure prediction horizon (seconds)')
    parser.add_argument('--label-width', type=int, default=30, help='Label window width (seconds)')
    parser.add_argument('--window-size', type=float, default=30.0, help='Feature window size (seconds)')
    parser.add_argument('--stride', type=float, default=5.0, help='Window stride (seconds)')
    parser.add_argument('--sampling-rate', type=int, default=256, help='Target sampling rate')
    parser.add_argument('--single-file', help='Process single ECG file')
    parser.add_argument('--eeg-file', help='Corresponding EEG file (for single file mode)')
    parser.add_argument('--annotation-file', help='Annotation file (for single file mode)')
    
    # Resume functionality
    parser.add_argument('--no-resume', action='store_true', 
                       help='Disable resume mode (reprocess all files)')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Force reprocessing of all files (ignore existing outputs)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actually processing')
    
    args = parser.parse_args()
    
    # Determine resume mode
    resume_mode = not (args.no_resume or args.force_reprocess)
    
    # Initialize pipeline
    pipeline = HRVFeaturePipeline(
        sampling_rate=args.sampling_rate,
        sph_seconds=args.sph,
        label_width_seconds=args.label_width,
        window_size_seconds=args.window_size,
        stride_seconds=args.stride,
        output_dir=args.output_dir,
        resume=resume_mode
    )
    
    # Show configuration
    print(f"Pipeline Configuration:")
    print(f"   SPH: {args.sph}s, Label width: {args.label_width}s")
    print(f"   Window: {args.window_size}s, Stride: {args.stride}s")
    print(f"   Output: {args.output_dir}")
    print(f"   Resume mode: {'enabled' if resume_mode else 'disabled'}")
    if args.dry_run:
        print(f"DRY RUN MODE - No files will be processed")
    print()
    
    if args.single_file:
        # Process single file
        if not args.eeg_file:
            print("Error: --eeg-file required for single file mode")
            sys.exit(1)
        
        features_df = pipeline.process_recording(
            eeg_file=args.eeg_file,
            ecg_file=args.single_file,
            annotation_file=args.annotation_file
        )
        
        if not features_df.empty:
            output_file = Path(args.output_dir) / f"{Path(args.single_file).stem}_features.csv"
            pipeline.save_features_csv(features_df, output_file)
        else:
            print("No features extracted")
    
    else:
        # Process entire dataset
        dataset_config = create_dataset_config(args.data_root)
        print(f"Found {len(dataset_config['recordings'])} recordings in dataset")
        
        if args.dry_run:
            # Dry run mode - show what would be processed
            print("DRY RUN - Checking which files need processing...")
            to_process = []
            to_skip = []
            
            for recording in dataset_config['recordings']:
                recording_id = recording['id']
                source_files = [
                    recording['eeg_file'],
                    recording['ecg_file'],
                    recording.get('annotation_file')
                ]
                
                if (resume_mode and 
                    pipeline._is_file_already_processed(recording_id, source_files) and 
                    pipeline._check_config_compatibility(recording_id)):
                    to_skip.append(recording_id)
                else:
                    to_process.append(recording_id)
            
            print(f"\nDRY RUN Summary:")
            print(f" Would skip: {len(to_skip)} recordings (already processed)")
            print(f"  Would process: {len(to_process)} recordings")
            
            if to_process:
                print(f"\nFiles to be processed:")
                for i, recording_id in enumerate(to_process, 1):
                    print(f"   {i:2d}. {recording_id}")
            
            if to_skip and len(to_skip) <= 10:
                print(f"\nFiles that would be skipped:")
                for i, recording_id in enumerate(to_skip, 1):
                    print(f"   {i:2d}. {recording_id}")
            elif to_skip:
                print(f"\nFiles that would be skipped: {len(to_skip)} files (use --no-resume to see list)")
            
            print(f"\nTo actually process, run the same command without --dry-run")
            
        else:
            # Actually process the dataset
            combined_df = pipeline.process_dataset(dataset_config)
            print(f"\nProcessing complete! Total windows: {len(combined_df)}")


if __name__ == "__main__":
    main()