#!/usr/bin/env python3
"""
Real-time Streaming Emulation for Seizure Prediction

This module emulates real-time inference by processing ECG data in windows,
maintaining a circular buffer of HRV features, and implementing debounce logic
for seizure prediction alarms.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import time
from typing import Dict, List, Tuple, Optional, Callable
from collections import deque
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from sklearn.ensemble import RandomForestClassifier
import pickle

from ecg_processing import ECGProcessor
from hrv_features import HRVFeatureExtractor
from labelers import FixedSPHLabeler
from data_processing_pipeline import AnnotationProcessor


class StreamingEmulator:
    """
    Real-time seizure prediction emulator.
    
    Processes ECG data in streaming fashion, maintains a circular buffer
    of HRV features, and implements debounce logic for alarm generation.
    """
    
    def __init__(self,
                 model_path: str,
                 seq_len: int = 36,
                 window_size_seconds: float = 30.0,
                 stride_seconds: float = 5.0,
                 sampling_rate: int = 256,
                 threshold: float = 0.5,
                 debounce_k: int = 3,
                 log_file: str = "stream_emulator.log"):
        """
        Initialize streaming emulator.
        
        Args:
            model_path: Path to trained model file
            seq_len: Sequence length for LSTM (default: 36)
            window_size_seconds: Window size for feature extraction (default: 30s)
            stride_seconds: Stride between windows (default: 5s)
            sampling_rate: ECG sampling rate (default: 256 Hz)
            threshold: Prediction threshold for alarm (default: 0.5)
            debounce_k: Number of consecutive positive predictions for alarm (default: 3)
            log_file: Path to log file for detection events
        """
        self.model_path = model_path
        self.seq_len = seq_len
        self.window_size_seconds = window_size_seconds
        self.stride_seconds = stride_seconds
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.debounce_k = debounce_k
        self.log_file = log_file
        
        # Initialize processing modules
        self.ecg_processor = ECGProcessor(sampling_rate=sampling_rate)
        self.hrv_extractor = HRVFeatureExtractor()
        
        # Circular buffer for HRV feature sequences
        self.feature_buffer = deque(maxlen=seq_len)
        self.feature_names = None
        
        # Debounce state
        self.consecutive_positives = 0
        self.alarm_active = False
        self.last_alarm_time = None
        
        # Detection logging
        self.detection_log = []
        
        # Load model
        self.model = self._load_model()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize feature statistics for normalization (if needed)
        self.feature_stats = None
    
    def _load_model(self):
        """Load trained model from file."""
        model_path = Path(self.model_path)
        
        if model_path.suffix == '.pkl':
            # Scikit-learn model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif model_path.suffix == '.h5' or model_path.suffix == '.keras':
            # Keras/TensorFlow model
            if not HAS_TENSORFLOW:
                raise ImportError("TensorFlow required for loading Keras models")
            model = keras.models.load_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")
        
        self.logger.info(f"Loaded model from {model_path}")
        return model
    
    def reset_state(self):
        """Reset emulator state for new recording."""
        self.feature_buffer.clear()
        self.consecutive_positives = 0
        self.alarm_active = False
        self.last_alarm_time = None
        self.detection_log = []
    
    def extract_window_features(self, ecg_segment: np.ndarray, 
                              window_start_time: float) -> Optional[np.ndarray]:
        """
        Extract HRV features from ECG window.
        
        Args:
            ecg_segment: ECG data for the window
            window_start_time: Start time of the window
            
        Returns:
            Feature vector or None if extraction failed
        """
        try:
            # Process ECG to get tachogram
            tachogram_result = self.ecg_processor.process_ecg_to_tachogram(ecg_segment)
            
            if len(tachogram_result['filtered_rr']) < 5:
                # Not enough beats for meaningful features
                return None
            
            # Extract HRV features
            features = self.hrv_extractor.compute_all_features(
                tachogram_result['filtered_rr'],
                tachogram_result['filtered_times']
            )
            
            # Convert to feature vector
            if self.feature_names is None:
                self.feature_names = self.hrv_extractor.get_feature_names()
            
            feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])
            
            # Handle NaN values
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            return feature_vector
            
        except Exception as e:
            self.logger.warning(f"Feature extraction failed at {window_start_time:.1f}s: {e}")
            return None
    
    def update_buffer_and_predict(self, feature_vector: np.ndarray, 
                                 current_time: float) -> Tuple[bool, float]:
        """
        Update feature buffer and make prediction.
        
        Args:
            feature_vector: New feature vector to add to buffer
            current_time: Current time in seconds
            
        Returns:
            Tuple of (prediction_made, probability)
        """
        # Add to buffer
        self.feature_buffer.append(feature_vector)
        
        # Check if we have enough features for prediction
        if len(self.feature_buffer) < self.seq_len:
            return False, 0.0
        
        # Prepare input for model
        if hasattr(self.model, 'predict'):
            # Scikit-learn or Keras model
            sequence = np.array(list(self.feature_buffer))
            
            if hasattr(self.model, 'predict_proba'):
                # Scikit-learn classifier
                X = sequence.flatten().reshape(1, -1)  # Flatten for sklearn
                probability = self.model.predict_proba(X)[0, 1]  # Probability of positive class
            else:
                # Keras model - expects 3D input (batch, timesteps, features)
                X = sequence.reshape(1, self.seq_len, -1)
                probability = float(self.model.predict(X)[0, 0])
        else:
            # Unknown model type
            self.logger.error("Unknown model type for prediction")
            return False, 0.0
        
        return True, probability
    
    def apply_debounce_logic(self, probability: float, current_time: float) -> Dict:
        """
        Apply debounce logic to determine if alarm should be triggered.
        
        Args:
            probability: Prediction probability
            current_time: Current time in seconds
            
        Returns:
            Dictionary with alarm information
        """
        # Check if prediction exceeds threshold
        is_positive = probability >= self.threshold
        
        if is_positive:
            self.consecutive_positives += 1
        else:
            self.consecutive_positives = 0
        
        # Determine if alarm should be triggered
        trigger_alarm = (self.consecutive_positives >= self.debounce_k and 
                        not self.alarm_active)
        
        alarm_info = {
            'probability': probability,
            'is_positive': is_positive,
            'consecutive_positives': self.consecutive_positives,
            'trigger_alarm': trigger_alarm,
            'alarm_active': self.alarm_active
        }
        
        if trigger_alarm:
            self.alarm_active = True
            self.last_alarm_time = current_time
            
            # Log alarm
            alarm_info['alarm_time'] = current_time
            self.logger.info(f"ALARM TRIGGERED at {current_time:.1f}s (probability: {probability:.3f})")
        
        return alarm_info
    
    def process_recording_streaming(self, ecg_data: np.ndarray, 
                                  seizure_times: List[Tuple[float, float]] = None) -> List[Dict]:
        """
        Process entire recording in streaming fashion.
        
        Args:
            ecg_data: Complete ECG signal
            seizure_times: List of (onset, end) times for seizures
            
        Returns:
            List of detection events and logs
        """
        self.reset_state()
        
        total_duration = len(ecg_data) / self.sampling_rate
        samples_per_window = int(self.window_size_seconds * self.sampling_rate)
        step_size = int(self.stride_seconds * self.sampling_rate)
        
        n_windows = (len(ecg_data) - samples_per_window) // step_size + 1
        
        self.logger.info(f"Processing {total_duration:.1f}s recording in {n_windows} windows")
        
        processing_log = []
        
        for i in range(n_windows):
            # Extract window
            start_sample = i * step_size
            end_sample = start_sample + samples_per_window
            current_time = end_sample / self.sampling_rate  # Use window end time
            
            ecg_window = ecg_data[start_sample:end_sample]
            
            # Extract features
            feature_vector = self.extract_window_features(ecg_window, current_time)
            
            if feature_vector is None:
                # Failed feature extraction - log and continue
                processing_log.append({
                    'time': current_time,
                    'window_idx': i,
                    'feature_extraction_success': False,
                    'prediction_made': False
                })
                continue
            
            # Make prediction
            prediction_made, probability = self.update_buffer_and_predict(feature_vector, current_time)
            
            if prediction_made:
                # Apply debounce logic
                alarm_info = self.apply_debounce_logic(probability, current_time)
                
                # Log detection event
                detection_event = {
                    'time': current_time,
                    'window_idx': i,
                    'feature_extraction_success': True,
                    'prediction_made': True,
                    **alarm_info
                }
                
                # Add seizure timing information if available
                if seizure_times:
                    detection_event['time_to_seizure'] = self._calculate_time_to_seizure(
                        current_time, seizure_times)
                
                processing_log.append(detection_event)
                self.detection_log.append(detection_event)
            else:
                # Not enough data for prediction yet
                processing_log.append({
                    'time': current_time,
                    'window_idx': i,
                    'feature_extraction_success': True,
                    'prediction_made': False,
                    'buffer_size': len(self.feature_buffer)
                })
        
        return processing_log
    
    def _calculate_time_to_seizure(self, current_time: float, 
                                 seizure_times: List[Tuple[float, float]]) -> Optional[float]:
        """Calculate time to next seizure onset."""
        for onset, end in seizure_times:
            if current_time < onset:
                return onset - current_time
        return None  # No future seizures
    
    def calculate_detection_metrics(self, processing_log: List[Dict], 
                                  seizure_times: List[Tuple[float, float]]) -> Dict:
        """
        Calculate detection performance metrics.
        
        Args:
            processing_log: Log of all processing events
            seizure_times: List of seizure (onset, end) times
            
        Returns:
            Dictionary with performance metrics
        """
        if not seizure_times:
            return {'no_seizures': True}
        
        # Find all alarm events
        alarms = [event for event in processing_log 
                 if event.get('trigger_alarm', False)]
        
        seizure_detection_results = []
        
        for onset, end in seizure_times:
            # Find alarms before this seizure
            pre_seizure_alarms = [alarm for alarm in alarms 
                                if alarm['time'] < onset]
            
            if pre_seizure_alarms:
                # Find the latest alarm before seizure
                latest_alarm = max(pre_seizure_alarms, key=lambda x: x['time'])
                warning_time = onset - latest_alarm['time']
                detected = True
                detection_time = latest_alarm['time']
            else:
                warning_time = None
                detected = False
                detection_time = None
            
            seizure_detection_results.append({
                'seizure_onset': onset,
                'seizure_end': end,
                'detected': detected,
                'detection_time': detection_time,
                'warning_time': warning_time
            })
        
        # Calculate summary metrics
        n_seizures = len(seizure_times)
        n_detected = sum(1 for result in seizure_detection_results if result['detected'])
        sensitivity = n_detected / n_seizures if n_seizures > 0 else 0.0
        
        # Warning time statistics
        warning_times = [result['warning_time'] for result in seizure_detection_results 
                        if result['warning_time'] is not None]
        
        mean_warning_time = np.mean(warning_times) if warning_times else 0.0
        std_warning_time = np.std(warning_times) if warning_times else 0.0
        
        # False alarm rate (alarms not followed by seizure within reasonable time)
        false_alarms = 0
        for alarm in alarms:
            # Check if alarm is followed by seizure within 300 seconds (5 minutes)
            is_true_alarm = any(onset - alarm['time'] <= 300 and onset > alarm['time'] 
                              for onset, end in seizure_times)
            if not is_true_alarm:
                false_alarms += 1
        
        # Estimate recording duration and calculate FPR/hour
        if processing_log:
            recording_duration_hours = max(event['time'] for event in processing_log) / 3600
            fpr_per_hour = false_alarms / recording_duration_hours if recording_duration_hours > 0 else 0.0
        else:
            fpr_per_hour = 0.0
        
        return {
            'n_seizures': n_seizures,
            'n_detected': n_detected,
            'sensitivity': sensitivity,
            'mean_warning_time': mean_warning_time,
            'std_warning_time': std_warning_time,
            'false_alarms': false_alarms,
            'fpr_per_hour': fpr_per_hour,
            'total_alarms': len(alarms),
            'seizure_results': seizure_detection_results
        }
    
    def save_log(self, processing_log: List[Dict], metrics: Dict, 
                output_file: str = None):
        """Save processing log and metrics to file."""
        if output_file is None:
            output_file = self.log_file
        
        log_data = {
            'emulator_config': {
                'model_path': self.model_path,
                'seq_len': self.seq_len,
                'window_size_seconds': self.window_size_seconds,
                'stride_seconds': self.stride_seconds,
                'sampling_rate': self.sampling_rate,
                'threshold': self.threshold,
                'debounce_k': self.debounce_k
            },
            'metrics': metrics,
            'processing_log': processing_log
        }
        
        with open(output_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.logger.info(f"Processing log saved to {output_file}")


def main():
    """Command-line interface for streaming emulation."""
    import argparse
    import mne
    
    parser = argparse.ArgumentParser(description='Run streaming seizure prediction emulation')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--ecg-file', required=True, help='ECG file to process')
    parser.add_argument('--annotation-file', help='Seizure annotation file')
    parser.add_argument('--output-log', default='stream_emulator.log', help='Output log file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('--debounce-k', type=int, default=3, help='Debounce parameter')
    parser.add_argument('--seq-len', type=int, default=36, help='Sequence length')
    
    args = parser.parse_args()
    
    # Load ECG data
    print(f"Loading ECG data from {args.ecg_file}...")
    raw_ecg = mne.io.read_raw_edf(args.ecg_file, preload=True, verbose=False)
    raw_ecg.resample(256, verbose=False)
    ecg_data = raw_ecg.get_data()[0]  # Assume single channel
    
    # Load seizure annotations if available
    seizure_times = []
    if args.annotation_file:
        processor = AnnotationProcessor()
        seizure_events = processor.load_annotations(args.annotation_file)
        
        for _, event in seizure_events.iterrows():
            onset = event['onset']
            duration = event.get('duration', 0)
            seizure_times.append((onset, onset + duration))
        
        print(f"Found {len(seizure_times)} seizure events")
    
    # Initialize emulator
    emulator = StreamingEmulator(
        model_path=args.model,
        seq_len=args.seq_len,
        threshold=args.threshold,
        debounce_k=args.debounce_k,
        log_file=args.output_log
    )
    
    # Run streaming emulation
    print("Running streaming emulation...")
    processing_log = emulator.process_recording_streaming(ecg_data, seizure_times)
    
    # Calculate metrics
    metrics = emulator.calculate_detection_metrics(processing_log, seizure_times)
    
    # Save results
    emulator.save_log(processing_log, metrics, args.output_log)
    
    # Print summary
    print("\n" + "="*50)
    print("STREAMING EMULATION RESULTS")
    print("="*50)
    
    if metrics.get('no_seizures'):
        print("No seizures in recording")
    else:
        print(f"Seizures detected: {metrics['n_detected']}/{metrics['n_seizures']} "
              f"(Sensitivity: {metrics['sensitivity']:.1%})")
        
        if metrics['n_detected'] > 0:
            print(f"Mean warning time: {metrics['mean_warning_time']:.1f} ± "
                  f"{metrics['std_warning_time']:.1f} seconds")
        
        print(f"False alarms: {metrics['false_alarms']}")
        print(f"False alarm rate: {metrics['fpr_per_hour']:.2f} per hour")
    
    print(f"\nDetailed log saved to: {args.output_log}")


if __name__ == "__main__":
    main()