#!/usr/bin/env python3
"""
Leave-One-Seizure-Out (LOSO) Evaluation Script

This module implements cross-validation for seizure prediction models using
leave-one-seizure-out or leave-one-recording-out strategies to avoid temporal leakage.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from lstm_sequences import LSTMSequenceBuilder


class LOSOEvaluator:
    """
    Leave-One-Seizure-Out evaluation for seizure prediction models.
    """
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize LOSO evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_feature_data(self, csv_files: List[str]) -> pd.DataFrame:
        """
        Load and combine feature data from multiple CSV files.
        
        Args:
            csv_files: List of paths to feature CSV files
            
        Returns:
            Combined DataFrame
        """
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df.sort_values(['subject_id', 'recording_id', 'window_center_time'])
    
    def identify_seizure_episodes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify individual seizure episodes for LOSO splits.
        
        Args:
            df: Features DataFrame
            
        Returns:
            DataFrame with seizure episode information
        """
        # Find seizure windows (label = 2)
        seizure_windows = df[df['label'] == 2].copy()
        
        if seizure_windows.empty:
            return pd.DataFrame(columns=['subject_id', 'recording_id', 'episode_id', 
                                       'start_time', 'end_time', 'duration'])
        
        episodes = []
        
        # Group by subject and recording
        for (subject, recording), group in seizure_windows.groupby(['subject_id', 'recording_id']):
            # Find continuous seizure periods
            times = group['window_center_time'].values
            
            # Identify gaps larger than expected window stride (indicating separate seizures)
            time_diffs = np.diff(times)
            gap_threshold = 30.0  # 30 seconds gap indicates separate seizure
            episode_breaks = np.where(time_diffs > gap_threshold)[0]
            
            # Split into episodes
            start_indices = [0] + list(episode_breaks + 1)
            end_indices = list(episode_breaks + 1) + [len(times)]
            
            for i, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices)):
                episode_times = times[start_idx:end_idx]
                episodes.append({
                    'subject_id': subject,
                    'recording_id': recording,
                    'episode_id': f"{subject}_{recording}_seizure_{i+1}",
                    'start_time': episode_times[0],
                    'end_time': episode_times[-1],
                    'duration': episode_times[-1] - episode_times[0]
                })
        
        return pd.DataFrame(episodes)
    
    def create_loso_splits(self, df: pd.DataFrame) -> List[Dict]:
        """
        Create leave-one-seizure-out cross-validation splits.
        
        Args:
            df: Features DataFrame
            
        Returns:
            List of train/test split dictionaries
        """
        seizure_episodes = self.identify_seizure_episodes(df)
        
        if seizure_episodes.empty:
            self.logger.warning("No seizure episodes found for LOSO splits")
            return []
        
        self.logger.info(f"Found {len(seizure_episodes)} seizure episodes for LOSO")
        
        splits = []
        
        for _, episode in seizure_episodes.iterrows():
            # Test set: all data from the recording containing this seizure
            test_mask = (df['subject_id'] == episode['subject_id']) & \
                       (df['recording_id'] == episode['recording_id'])
            
            # Training set: all other recordings
            train_mask = ~test_mask
            
            # Additional constraint: remove pre-seizure windows from test set
            # that might be too close to other seizures in training set
            # (This helps prevent information leakage)
            
            split_info = {
                'episode_id': episode['episode_id'],
                'test_subject': episode['subject_id'],
                'test_recording': episode['recording_id'],
                'train_mask': train_mask,
                'test_mask': test_mask,
                'seizure_start': episode['start_time'],
                'seizure_end': episode['end_time']
            }
            
            splits.append(split_info)
        
        return splits
    
    def extract_features_and_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract feature matrix and labels from DataFrame.
        
        Args:
            df: Features DataFrame
            
        Returns:
            Tuple of (features, labels, feature_names)
        """
        # Identify feature columns
        metadata_cols = ['subject_id', 'recording_id', 'window_start_time', 
                        'window_center_time', 'window_end_time', 'label']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        features = df[feature_cols].values
        labels = df['label'].values
        
        # Handle NaN values
        nan_mask = np.isnan(features).any(axis=1)
        if np.any(nan_mask):
            self.logger.warning(f"Replacing {np.sum(nan_mask)} NaN features with zeros")
            features[nan_mask] = 0.0
        
        return features, labels, feature_cols
    
    def evaluate_split(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      model_type: str = 'rf') -> Dict:
        """
        Evaluate a single train/test split.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            model_type: Model type ('rf', 'lr', 'lstm')
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42, 
                                         class_weight='balanced')
        elif model_type == 'lr':
            model = LogisticRegression(random_state=42, class_weight='balanced',
                                     max_iter=1000)
        elif model_type == 'lstm':
            if not HAS_TENSORFLOW:
                raise ImportError("TensorFlow required for LSTM models")
            model = self._build_lstm_model(X_train.shape[1])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        if model_type == 'lstm':
            # Reshape for LSTM if needed
            if len(X_train.shape) == 2:
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0,
                     validation_split=0.2)
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        else:
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        return metrics
    
    def _build_lstm_model(self, n_features: int) -> keras.Model:
        """Build simple LSTM model for evaluation."""
        model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True, input_shape=(None, n_features)),
            keras.layers.Dropout(0.3),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict:
        """Calculate evaluation metrics."""
        # Handle multi-class to binary conversion
        y_true_binary = (y_true > 0).astype(int)  # Any non-normal class as positive
        y_pred_binary = (y_pred > 0).astype(int)
        
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_true_binary, y_pred_proba)
        except:
            roc_auc = 0.0
        
        # False positive rate per hour (assuming 5-second windows)
        n_negative_windows = np.sum(y_true_binary == 0)
        recording_hours = (n_negative_windows * 5) / 3600  # Convert windows to hours
        fpr_per_hour = fp / recording_hours if recording_hours > 0 else 0.0
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'roc_auc': roc_auc,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'fpr_per_hour': fpr_per_hour
        }
    
    def run_loso_evaluation(self, csv_files: List[str], 
                           model_type: str = 'rf') -> pd.DataFrame:
        """
        Run complete LOSO evaluation.
        
        Args:
            csv_files: List of feature CSV files
            model_type: Model type to evaluate
            
        Returns:
            DataFrame with per-fold results
        """
        self.logger.info(f"Starting LOSO evaluation with {model_type} model")
        
        # Load data
        df = self.load_feature_data(csv_files)
        self.logger.info(f"Loaded {len(df)} windows from {len(csv_files)} files")
        
        # Create splits
        splits = self.create_loso_splits(df)
        self.logger.info(f"Created {len(splits)} LOSO splits")
        
        # Extract features
        X, y, feature_names = self.extract_features_and_labels(df)
        
        # Evaluate each split
        results = []
        
        for i, split in enumerate(splits):
            self.logger.info(f"Evaluating fold {i+1}/{len(splits)}: {split['episode_id']}")
            
            # Get train/test data
            X_train = X[split['train_mask']]
            y_train = y[split['train_mask']]
            X_test = X[split['test_mask']]
            y_test = y[split['test_mask']]
            
            # Check if we have both classes in training set
            if len(np.unique(y_train)) < 2:
                self.logger.warning(f"Fold {i+1}: Insufficient class diversity in training set")
                continue
            
            # Evaluate
            metrics = self.evaluate_split(X_train, y_train, X_test, y_test, model_type)
            
            # Add fold information
            result = {
                'fold': i + 1,
                'episode_id': split['episode_id'],
                'test_subject': split['test_subject'],
                'test_recording': split['test_recording'],
                'n_train': len(X_train),
                'n_test': len(X_test),
                'train_pos_rate': np.mean(y_train > 0),
                'test_pos_rate': np.mean(y_test > 0),
                **metrics
            }
            
            results.append(result)
            
            self.logger.info(f"Fold {i+1} - Sensitivity: {metrics['sensitivity']:.3f}, "
                           f"FPR/h: {metrics['fpr_per_hour']:.2f}")
        
        results_df = pd.DataFrame(results)
        
        # Save results
        results_file = self.output_dir / f"loso_results_{model_type}.csv"
        results_df.to_csv(results_file, index=False)
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(results_df)
        
        summary_file = self.output_dir / f"loso_summary_{model_type}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"LOSO evaluation complete. Results saved to {results_file}")
        self.logger.info(f"Summary statistics saved to {summary_file}")
        
        return results_df
    
    def _calculate_summary_stats(self, results_df: pd.DataFrame) -> Dict:
        """Calculate summary statistics across all folds."""
        numeric_cols = ['sensitivity', 'specificity', 'precision', 'roc_auc', 'fpr_per_hour']
        
        summary = {
            'n_folds': len(results_df),
            'mean_metrics': {},
            'std_metrics': {},
            'median_metrics': {}
        }
        
        for col in numeric_cols:
            if col in results_df.columns:
                values = results_df[col].dropna()
                summary['mean_metrics'][col] = float(values.mean())
                summary['std_metrics'][col] = float(values.std())
                summary['median_metrics'][col] = float(values.median())
        
        # Additional summary statistics
        summary['seizures_detected'] = int(results_df['sensitivity'].gt(0).sum())
        summary['perfect_sensitivity_folds'] = int(results_df['sensitivity'].eq(1.0).sum())
        
        return summary


def main():
    """Command-line interface for LOSO evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run LOSO evaluation for seizure prediction')
    parser.add_argument('--input-dir', required=True, help='Directory with feature CSV files')
    parser.add_argument('--output-dir', default='evaluation_results', help='Output directory')
    parser.add_argument('--model', choices=['rf', 'lr', 'lstm'], default='rf', 
                       help='Model type to evaluate')
    
    args = parser.parse_args()
    
    # Find CSV files
    input_dir = Path(args.input_dir)
    csv_files = list(input_dir.glob("*_features.csv"))
    
    if not csv_files:
        print(f"No feature CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Run evaluation
    evaluator = LOSOEvaluator(output_dir=args.output_dir)
    results_df = evaluator.run_loso_evaluation(
        [str(f) for f in csv_files], 
        model_type=args.model
    )
    
    # Print summary
    if not results_df.empty:
        print("\n" + "="*50)
        print("LOSO EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {args.model}")
        print(f"Number of folds: {len(results_df)}")
        print(f"Mean sensitivity: {results_df['sensitivity'].mean():.3f} ± {results_df['sensitivity'].std():.3f}")
        print(f"Mean FPR/hour: {results_df['fpr_per_hour'].mean():.2f} ± {results_df['fpr_per_hour'].std():.2f}")
        print(f"Mean ROC AUC: {results_df['roc_auc'].mean():.3f} ± {results_df['roc_auc'].std():.3f}")


if __name__ == "__main__":
    main()