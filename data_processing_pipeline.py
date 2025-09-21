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
import tempfile
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from urllib.parse import urlparse
import io
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor            

def process_single_run_worker(run_data: Dict, config: Dict) -> Optional[Dict]:
    """Worker function for parallel processing of a single run."""
    try:
        # Initialize processors in worker process
        annotation_processor = AnnotationProcessor()
        hrv_processor = HRVFeatureProcessor(
            sampling_rate=config['sampling_rate'],
            sph_seconds=config['sph_seconds'],
            label_width_seconds=config['label_width_seconds'],
            window_size_seconds=config['window_size_seconds'],
            stride_seconds=config['stride_seconds']
        )
        s3_handler = S3FileHandler()
        
        # Check if we have required files
        if not run_data['eeg_file'] or not run_data['ecg_file']:
            return None
        
        # Create output filename
        output_filename = f"{run_data['subject']}_{run_data['session']}_run-{run_data['run']}_features.csv"
        
        if s3_handler.is_s3_path(config['output_dir']):
            output_file = f"{config['output_dir']}/{output_filename}"
        else:
            output_file = Path(config['output_dir']) / output_filename
        
        # Load seizure annotations
        seizure_events = pd.DataFrame()
        if run_data['annotation_file']:
            seizure_events = annotation_processor.load_annotations(run_data['annotation_file'])
        
        # Process recording to extract HRV features
        features_df = hrv_processor.process_recording(
            run_data['eeg_file'], run_data['ecg_file'], seizure_events
        )
        
        if features_df.empty:
            return None
        
        # Save CSV file (S3 or local)
        if s3_handler.is_s3_path(config['output_dir']):
            s3_handler.upload_dataframe_to_s3(features_df, output_file)
        else:
            features_df.to_csv(output_file, index=False)
        
        # Calculate statistics
        label_counts = features_df['label'].value_counts().sort_index()
        total_windows = len(features_df)
        
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
        print(f"Worker error for {run_data['subject']}/{run_data['session']}/run-{run_data['run']}: {e}")
        return None

def main(): as_completed
import time
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Import our specialized modules
from labelers import FixedSPHLabeler, WindowCenterLabeler
from ecg_processing import ECGProcessor
from hrv_features import HRVFeatureExtractor

class S3FileHandler:
    """Handler for S3 file operations with local file fallback."""
    
    def __init__(self):
        """Initialize S3 client."""
        try:
            self.s3_client = boto3.client('s3')
            # Test S3 connection
            #self.s3_client.list_buckets()
            self.s3_available = True
            print("S3 connection established successfully")
        except (NoCredentialsError, ClientError) as e:
            print(f"Warning: S3 not available ({e}). Will use local files only.")
            self.s3_client = None
            self.s3_available = False
    
    def is_s3_path(self, path: str) -> bool:
        """Check if path is an S3 URL."""
        return str(path).startswith('s3://')
    
    def parse_s3_path(self, s3_path: str) -> Tuple[str, str]:
        """Parse S3 path into bucket and key."""
        parsed = urlparse(s3_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        return bucket, key
    
    def list_s3_objects(self, s3_path: str, suffix: str = "") -> List[str]:
        """List objects in S3 bucket with optional suffix filter."""
        if not self.s3_available:
            raise RuntimeError("S3 not available")
        
        bucket, prefix = self.parse_s3_path(s3_path)
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            
            objects = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if suffix and not key.endswith(suffix):
                            continue
                        objects.append(f"s3://{bucket}/{key}")
            
            return sorted(objects)
            
        except ClientError as e:
            print(f"Error listing S3 objects: {e}")
            return []
    
    def download_s3_file(self, s3_path: str) -> str:
        """Download S3 file to temporary local file and return local path."""
        if not self.s3_available:
            raise RuntimeError("S3 not available")
        
        bucket, key = self.parse_s3_path(s3_path)
        
        # Create temporary file with same extension
        suffix = Path(key).suffix
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            self.s3_client.download_file(bucket, key, temp_path)
            return temp_path
        except ClientError as e:
            print(f"Error downloading {s3_path}: {e}")
            os.unlink(temp_path)  # Clean up temp file
            raise
    
    def batch_download_s3_files(self, s3_paths: List[str]) -> Dict[str, str]:
        """Download multiple S3 files concurrently."""
        if not self.s3_available:
            raise RuntimeError("S3 not available")
        
        local_files = {}
        max_workers = min(len(s3_paths), 4)  # Limit concurrent downloads
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_s3path = {
                executor.submit(self.download_s3_file, s3_path): s3_path 
                for s3_path in s3_paths
            }
            
            for future in as_completed(future_to_s3path):
                s3_path = future_to_s3path[future]
                try:
                    local_files[s3_path] = future.result()
                except Exception as e:
                    print(f"Failed to download {s3_path}: {e}")
        
        return local_files
    
    def upload_s3_file(self, local_path: str, s3_path: str):
        """Upload local file to S3."""
        if not self.s3_available:
            raise RuntimeError("S3 not available")
        
        bucket, key = self.parse_s3_path(s3_path)
        
        try:
            self.s3_client.upload_file(local_path, bucket, key)
            print(f"Uploaded {local_path} to {s3_path}")
        except ClientError as e:
            print(f"Error uploading to {s3_path}: {e}")
            raise
    
    def upload_dataframe_to_s3(self, df: pd.DataFrame, s3_path: str):
        """Upload DataFrame directly to S3 as CSV."""
        if not self.s3_available:
            raise RuntimeError("S3 not available")
        
        bucket, key = self.parse_s3_path(s3_path)
        
        try:
            # Convert DataFrame to CSV string
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=csv_string.encode('utf-8'),
                ContentType='text/csv'
            )
            print(f"Uploaded DataFrame to {s3_path}")
        except ClientError as e:
            print(f"Error uploading DataFrame to {s3_path}: {e}")
            raise
    
    def upload_json_to_s3(self, data: dict, s3_path: str):
        """Upload dictionary as JSON to S3."""
        if not self.s3_available:
            raise RuntimeError("S3 not available")
        
        bucket, key = self.parse_s3_path(s3_path)
        
        try:
            json_string = json.dumps(data, indent=2)
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json_string.encode('utf-8'),
                ContentType='application/json'
            )
            print(f"Uploaded JSON to {s3_path}")
        except ClientError as e:
            print(f"Error uploading JSON to {s3_path}: {e}")
            raise

class DataDiscovery:
    """Module for discovering and organizing dataset files."""
    
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.subjects_data = {}
        self.s3_handler = S3FileHandler()
        
    def scan_dataset(self) -> Dict:
        """Scan the dataset and organize files by subject/session/run."""
        print("Scanning dataset structure...")
        
        if self.s3_handler.is_s3_path(self.data_root):
            return self._scan_s3_dataset()
        else:
            return self._scan_local_dataset()
    
    def _scan_local_dataset(self) -> Dict:
        """Scan local filesystem dataset."""
        # Find all subject directories
        subject_dirs = sorted(glob.glob(str(Path(self.data_root) / "sub-*")))
        
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
    
    def _scan_s3_dataset(self) -> Dict:
        """Scan S3 dataset."""
        print("Scanning S3 dataset...")
        
        # List all objects in the S3 bucket
        all_objects = self.s3_handler.list_s3_objects(self.data_root)
        
        # Organize by subject/session
        for s3_path in all_objects:
            # Extract path components
            # Example: s3://seizury-data/ds005873/sub-001/ses-01/eeg/sub-001_ses-01_task-szMonitoring_run-01_eeg.edf
            path_parts = s3_path.replace(self.data_root + "/", "").split("/")
            
            if len(path_parts) < 4:  # Need at least sub-XXX/ses-XX/modality/file
                continue
                
            subject_id = path_parts[0]  # sub-001
            session_id = path_parts[1]  # ses-01
            modality = path_parts[2]    # eeg or ecg
            filename = path_parts[3]    # actual filename
            
            # Initialize subject if not exists
            if subject_id not in self.subjects_data:
                self.subjects_data[subject_id] = {}
            
            # Initialize session if not exists
            if session_id not in self.subjects_data[subject_id]:
                self.subjects_data[subject_id][session_id] = {
                    'eeg_files': [],
                    'ecg_files': [],
                    'annotation_files': []
                }
            
            # Categorize files by type
            if modality == "eeg":
                if filename.endswith("_eeg.edf"):
                    self.subjects_data[subject_id][session_id]['eeg_files'].append(s3_path)
                elif filename.endswith("_events.tsv"):
                    self.subjects_data[subject_id][session_id]['annotation_files'].append(s3_path)
            elif modality == "ecg":
                if filename.endswith("_ecg.edf"):
                    self.subjects_data[subject_id][session_id]['ecg_files'].append(s3_path)
        
        # Sort file lists
        for subject_id in self.subjects_data:
            for session_id in self.subjects_data[subject_id]:
                self.subjects_data[subject_id][session_id]['eeg_files'].sort()
                self.subjects_data[subject_id][session_id]['ecg_files'].sort()
                self.subjects_data[subject_id][session_id]['annotation_files'].sort()
        
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
        self.s3_handler = S3FileHandler()
    
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
        local_file = None
        
        try:
            # Handle S3 files
            if self.s3_handler.is_s3_path(annotation_file):
                if not self.s3_handler.s3_available:
                    print(f"Warning: S3 not available, cannot load {annotation_file}")
                    return pd.DataFrame()
                local_file = self.s3_handler.download_s3_file(annotation_file)
                file_to_read = local_file
            else:
                # Local file
                if not os.path.exists(annotation_file):
                    return pd.DataFrame()
                file_to_read = annotation_file
            
            annotations = pd.read_csv(file_to_read, sep='\t')
            
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
        finally:
            # Clean up temporary file if downloaded from S3
            if local_file and os.path.exists(local_file):
                os.unlink(local_file)

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
        self.s3_handler = S3FileHandler()
        
    def process_recording(self, eeg_file: str, ecg_file: str, 
                         seizure_events: pd.DataFrame) -> pd.DataFrame:
        """Process a recording to extract HRV features with Fixed SPH labels."""
        
        eeg_local_file = None
        ecg_local_file = None
        
        try:
            # Handle S3 files - batch download if both are S3
            s3_files_to_download = []
            if self.s3_handler.is_s3_path(eeg_file):
                s3_files_to_download.append(eeg_file)
            if self.s3_handler.is_s3_path(ecg_file):
                s3_files_to_download.append(ecg_file)
            
            if s3_files_to_download:
                # Batch download for efficiency
                downloaded_files = self.s3_handler.batch_download_s3_files(s3_files_to_download)
                eeg_file_to_read = downloaded_files.get(eeg_file, eeg_file)
                ecg_file_to_read = downloaded_files.get(ecg_file, ecg_file)
                
                # Track temp files for cleanup
                if eeg_file in downloaded_files:
                    eeg_local_file = downloaded_files[eeg_file]
                if ecg_file in downloaded_files:
                    ecg_local_file = downloaded_files[ecg_file]
            else:
                # Local files
                eeg_file_to_read = eeg_file
                ecg_file_to_read = ecg_file
            
            # Load EEG with minimal processing
            raw_eeg = mne.io.read_raw_edf(eeg_file_to_read, preload=True, verbose=False)
            if raw_eeg.info['sfreq'] != self.sampling_rate:
                raw_eeg.resample(self.sampling_rate, verbose=False)
            
            # Create Fixed SPH labels
            labels = self.labeler.create_labels(raw_eeg, seizure_events)
            
            # Load ECG with minimal processing
            raw_ecg = mne.io.read_raw_edf(ecg_file_to_read, preload=True, verbose=False)
            if raw_ecg.info['sfreq'] != self.sampling_rate:
                raw_ecg.resample(self.sampling_rate, verbose=False)
            ecg_data = raw_ecg.get_data()[0]  # Assume single channel
            
            # Clear raw objects to free memory
            del raw_eeg, raw_ecg
            
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
                tachogram_result, labels, min_length / self.sampling_rate
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
            
        finally:
            # Clean up temporary files
            if eeg_local_file and os.path.exists(eeg_local_file):
                os.unlink(eeg_local_file)
            if ecg_local_file and os.path.exists(ecg_local_file):
                os.unlink(ecg_local_file)
    
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
    
    def __init__(self, data_root: str, output_dir: str = "hrv_features", 
                 n_workers: int = None, use_parallel: bool = True):
        self.data_root = data_root
        self.output_dir = output_dir
        self.s3_handler = S3FileHandler()
        self.use_parallel = use_parallel
        
        # Set workers based on system capabilities
        if n_workers is None:
            cpu_count = multiprocessing.cpu_count()
            # For m7i-flex.large (2 vCPUs), use both cores but leave headroom
            if cpu_count <= 2:
                self.n_workers = 2
            else:
                self.n_workers = min(cpu_count - 1, 4)  # Max 4 workers for memory constraints
        else:
            self.n_workers = n_workers
            
        print(f"Configured for {self.n_workers} parallel workers")
        
        # Create output directory only if it's local
        if not self.s3_handler.is_s3_path(output_dir):
            Path(output_dir).mkdir(exist_ok=True)
        
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
        print(f"  Parallel workers: {self.n_workers}")
        
        # Step 1: Discover data
        start_time = time.time()
        self.discovery.scan_dataset()
        self.discovery.print_summary()
        
        matched_runs = self.discovery.match_runs()
        print(f"\nFound {len(matched_runs)} matched runs to process")
        discovery_time = time.time() - start_time
        print(f"Discovery completed in {discovery_time:.2f}s")
        
        # Step 2: Process runs (parallel or sequential)
        if self.use_parallel and self.n_workers > 1:
            self._process_parallel(matched_runs)
        else:
            self._process_sequential(matched_runs)
        
        # Step 3: Save comprehensive results
        self._save_results()
    
    def _process_parallel(self, matched_runs: List[Dict]):
        """Process runs using parallel workers."""
        print(f"\nStarting parallel processing with {self.n_workers} workers...")
        
        # Create serializable config for workers
        worker_config = {
            'output_dir': self.output_dir,
            'sampling_rate': self.hrv_processor.sampling_rate,
            'sph_seconds': self.hrv_processor.sph_seconds,
            'label_width_seconds': self.hrv_processor.label_width_seconds,
            'window_size_seconds': self.hrv_processor.window_size_seconds,
            'stride_seconds': self.hrv_processor.stride_seconds
        }
        
        start_time = time.time()
        completed = 0
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all jobs
            future_to_run = {
                executor.submit(process_single_run_worker, run_data, worker_config): (i, run_data)
                for i, run_data in enumerate(matched_runs)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_run):
                i, run_data = future_to_run[future]
                completed += 1
                
                try:
                    result = future.result(timeout=600)  # 10 min timeout per file
                    if result:
                        self.processing_results.append(result)
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (len(matched_runs) - completed) / rate if rate > 0 else 0
                        print(f"Completed {completed}/{len(matched_runs)} "
                              f"({completed/len(matched_runs)*100:.1f}%) "
                              f"- Rate: {rate:.2f}/min - ETA: {eta/60:.1f}min")
                    else:
                        print(f"No features extracted for {run_data['subject']}/{run_data['session']}/run-{run_data['run']}")
                        
                except Exception as e:
                    print(f"Failed {run_data['subject']}/{run_data['session']}/run-{run_data['run']}: {e}")
    
    def _process_sequential(self, matched_runs: List[Dict]):
        """Process runs sequentially (fallback method)."""
        print("\nProcessing sequentially...")
        
        start_time = time.time()
        
        for i, run_data in enumerate(matched_runs):
            print(f"\nProcessing run {i+1}/{len(matched_runs)}: {run_data['subject']}/{run_data['session']}/run-{run_data['run']}")
            
            result = self._process_single_run(run_data)
            if result:
                self.processing_results.append(result)
                
            # Progress reporting
            if (i + 1) % 10 == 0 or i == len(matched_runs) - 1:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(matched_runs) - i - 1) / rate if rate > 0 else 0
                print(f"Progress: {i+1}/{len(matched_runs)} ({(i+1)/len(matched_runs)*100:.1f}%) "
                      f"- Rate: {rate*60:.1f}/min - ETA: {eta/60:.1f}min")
        
    def _process_single_run(self, run_data: Dict) -> Optional[Dict]:
        """Process a single run to extract HRV features."""
        try:
            # Check if we have required files
            if not run_data['eeg_file'] or not run_data['ecg_file']:
                print(f"  Skipping - missing EEG or ECG file")
                return None
            
            # Create output filename
            output_filename = f"{run_data['subject']}_{run_data['session']}_run-{run_data['run']}_features.csv"
            
            if self.s3_handler.is_s3_path(self.output_dir):
                output_file = f"{self.output_dir}/{output_filename}"
            else:
                output_file = Path(self.output_dir) / output_filename
            
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
            
            # Save CSV file (S3 or local)
            if self.s3_handler.is_s3_path(self.output_dir):
                self.s3_handler.upload_dataframe_to_s3(features_df, output_file)
            else:
                features_df.to_csv(output_file, index=False)
            
            # Calculate statistics
            label_counts = features_df['label'].value_counts().sort_index()
            total_windows = len(features_df)
            
            print(f"  Created {total_windows} windows")
            print(f"  Label distribution: {dict(label_counts)}")
            print(f"  Saved to: {output_filename}")
            
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
        if self.s3_handler.is_s3_path(self.output_dir):
            summary_file = f"{self.output_dir}/processing_summary.csv"
            self.s3_handler.upload_dataframe_to_s3(summary_df, summary_file)
        else:
            summary_file = Path(self.output_dir) / "processing_summary.csv"
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

        if self.s3_handler.is_s3_path(self.output_dir):
            info_file = f"{self.output_dir}/dataset_info.json"
            self.s3_handler.upload_json_to_s3(dataset_info, info_file)
        else:
            info_file = Path(self.output_dir) / "dataset_info.json"
            with open(info_file, 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
        print(f"Dataset info saved to: {info_file}")
        print(f"CSV feature files saved in: {self.output_dir}")
        
        # Create combined CSV
        print(f"\nCreating combined features CSV...")
        if self.s3_handler.is_s3_path(self.output_dir):
            # For S3, we need to list and download feature files to combine them
            print("Note: Combined CSV creation for S3 output not implemented in this version.")
            print("Individual feature files are available in S3.")
        else:
            # Local file combining (existing logic)
            all_csvs = list(Path(self.output_dir).glob("*_features.csv"))
            if all_csvs:
                combined_dfs = []
                for csv_file in all_csvs:
                    df = pd.read_csv(csv_file)
                    combined_dfs.append(df)
                
                combined_df = pd.concat(combined_dfs, ignore_index=True)
                combined_file = Path(self.output_dir) / "combined_features.csv"
                combined_df.to_csv(combined_file, index=False)
                print(f"Combined features saved to: {combined_file}")
                print(f"Total combined windows: {len(combined_df):,}")

def main():
    """Main function to run the HRV feature extraction pipeline."""
    
    # Configuration - LOCAL FILES (comment out if using S3)
    #data_root = "/Volumes/Seizury/ds005873"
    #output_dir = "/Volumes/Seizury/HRV/hrv_features"

    # Configuration - AWS S3 (comment out if using local files)
    data_root = "s3://seizury-data/ds005873"
    output_dir = "s3://seizury-data/hrv_features"

    # Performance settings for m7i-flex.large (2 vCPUs, 8GB RAM)
    n_workers = 2  # Use both CPU cores
    use_parallel = True  # Enable parallel processing
    
    print(f"Data source: {data_root}")
    print(f"Output destination: {output_dir}")
    print(f"Parallel processing: {use_parallel} ({n_workers} workers)")
    
    # Create and run pipeline
    pipeline = DataProcessingPipeline(
        data_root=data_root, 
        output_dir=output_dir,
        n_workers=n_workers,
        use_parallel=use_parallel
    )
    
    start_time = time.time()
    pipeline.process_dataset()
    total_time = time.time() - start_time
    
    print(f"\nTotal processing time: {total_time/60:.2f} minutes")
    if pipeline.processing_results:
        rate = len(pipeline.processing_results) / (total_time / 60)
        print(f"Processing rate: {rate:.2f} files/minute")

if __name__ == "__main__":
    # Required for multiprocessing on Windows and some Unix systems
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Method already set
    main()