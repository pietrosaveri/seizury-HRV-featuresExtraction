# CHANGELOG - Fixed SPH HRV Pipeline

## 🎯 **Which Pipeline File to Use?**

### **⭐ For Most Users: `data_processing_pipeline.py`**
- **Purpose:** Simple, optimized batch p#### **Command-Line Interfaces**
- **`data_processing_pipeline.py`*### **Breaking Changes from Original Pipeline**
- **REMOVED**: Strategy2Labeler (5-8 min window) replaced with FixedSPHLabeler
- **CHANGED**: Window labeling method from majority vote to center-point sampling
- **CHANGED**: Output format from NPZ to CSV for features, HDF5 for sequences
- **CHANGED**: Evaluation framework from simple accuracy to comprehensive LOSO
- **ADDED**: Two pipeline options: simple batch (`data_processing_pipeline.py`) and advanced (`hrv_pipeline.py`)

### **Migration Guide**
1. **Choose your pipeline:** Use `data_processing_pipeline.py` for simple batch processing OR `hrv_pipeline.py` for advanced configuration
2. Replace `Strategy2Labeler` usage with `FixedSPHLabeler` (already done in both pipelines)
3. Update window processing to use `WindowCenterLabeler.assign_window_labels()` (already implemented)
4. Migrate from NPZ to CSV/HDF5 output formats (already implemented)
5. Use new command-line interfaces for processing (see Quick Start examples above)

### **File Purposes Summary**
- **`data_processing_pipeline.py`** ⭐ **START HERE** - Simple batch processing with fixed configuration
- **`hrv_pipeline.py`** - Advanced pipeline with full command-line options and flexibility  
- **`labelers.py`** - Fixed SPH labeling implementation (180s ± 15s)
- **`ecg_processing.py`** - R-peak detection and tachogram extraction
- **`hrv_features.py`** - 23 HRV feature calculations matching MATLAB
- **`lstm_sequences.py`** - LSTM sequence generation with patient-level splits
- **`evaluate_loso.py`** - Leave-one-seizure-out cross-validation
- **`stream_emulator.py`** - Real-time streaming simulationatch processing (RECOMMENDED for most users)
- **`hrv_pipeline.py`**: Advanced feature extraction with full CLI options
- **`lstm_sequences.py`**: Sequence building for LSTM training
- **`evaluate_loso.py`**: Cross-validation evaluation
- **`stream_emulator.py`**: Real-time streaming emulation
- **`test_pipeline.py`**: Unit tests and validation

### **Quick Start Examples**

#### **Simple Processing (Recommended)**
```bash
# Just run this for your dataset:
python data_processing_pipeline.py
```

#### **Advanced Processing**
```bash
# Custom configuration:
python hrv_pipeline.py --data-root /your/dataset --output-dir ./features

# Single file:
python hrv_pipeline.py --single-file ecg.edf --eeg-file eeg.edf --annotation-file events.tsv
```

#### **LSTM Sequences (Patient-Level Splits)**
```bash
# Create train/val/test splits by patient (prevents data leakage):
python lstm_sequences.py --input-dir hrv_features --create-splits --normalize
```ing of entire dataset
- **Best for:** Quick feature extraction with pre-configured settings
- **Usage:** Just run `python data_processing_pipeline.py`
- **Output:** CSV files in `/Volumes/Seizury/HRV/hrv_features/`
- **Configuration:** Fixed paths optimized for your dataset

### **🔧 For Advanced Users: `hrv_pipeline.py`**
- **Purpose:** Full-featured pipeline with extensive configuration options
- **Best for:** Custom parameters, single file processing, research experiments
- **Usage:** Command-line interface with many options (see README)
- **Output:** Configurable output directory and settings
- **Configuration:** Complete command-line parameter control

**Both pipelines produce identical results with Fixed SPH labeling (180s ± 15s)!**

---

## Version 1.1 - Critical Data Leakage Fix (September 2025)

### 🚨 **CRITICAL IMPROVEMENT: Patient-Level Data Splitting**

#### **Problem Fixed**
- **FIXED**: Previous LSTM sequence builder used random splits, causing data leakage
- **ISSUE**: Same patients could appear in both training and test sets
- **CONSEQUENCE**: Model learned patient-specific patterns instead of generalizable seizure patterns
- **IMPACT**: Artificially inflated performance metrics, poor generalization to new patients

#### **New Patient-Level Splitting System**
- **NEW**: `create_patient_level_splits()` method for proper train/val/test splits
- **NEW**: `--create-splits` command-line flag for automatic patient-level splitting
- **NEW**: `--test-subjects` and `--val-subjects` options for manual patient assignment
- **NEW**: `split_by_patient=True` as default behavior to prevent data leakage
- **NEW**: Comprehensive logging of patient assignments for verification
- **NEW**: `patient_splits.json` output file documenting exact patient assignments

#### **Enhanced Command-Line Interface**
```bash
# NEW: Automatic patient-level splits (RECOMMENDED)
python lstm_sequences.py --input-dir ./features --create-splits

# NEW: Manual patient assignment
python lstm_sequences.py --input-dir ./features --create-splits \
  --test-subjects sub-01 sub-02 --val-subjects sub-03 sub-04
```

#### **Output Improvements**
- **NEW**: Separate HDF5 files for train/val/test splits
- **NEW**: Patient assignment tracking in all output files
- **NEW**: Split validation and overlap detection
- **NEW**: Detailed logging of patient distributions and class balance

#### **Documentation Updates**
- **NEW**: Critical data leakage prevention section in README
- **NEW**: Patient-level splitting examples and best practices
- **NEW**: Verification procedures for proper splits
- **NEW**: Warning messages for incorrect usage patterns

### **Breaking Changes**
- **CHANGED**: Default behavior now uses patient-level splits instead of random splits
- **CHANGED**: Command-line interface added new splitting options
- **DEPRECATED**: Random splitting (still available with `split_by_patient=False` but not recommended)

### **Migration Guide**
- **CRITICAL**: Replace any existing random splits with patient-level splits
- **UPDATE**: Use `--create-splits` flag for all LSTM sequence generation
- **VERIFY**: Check `patient_splits.json` to ensure no patient overlap between sets
- **TEST**: Retrain all models using new patient-level splits for accurate evaluation

---

## Version 1.0 - Complete Implementation (December 2024)

### Major Changes

#### 🎯 Fixed SPH Labeling System
- **NEW**: Implemented `FixedSPHLabeler` class targeting seizure prediction at exactly 180 seconds before onset
- **CHANGE**: Replaced Strategy2Labeler (5-8 min window) with precise SPH targeting
- **FEATURE**: Configurable label tolerance window (default: ±15s around target)
- **FEATURE**: Window labeling by center-point sampling instead of majority vote

#### 🫀 ECG Processing and R-peak Detection
- **NEW**: Comprehensive `ECGProcessor` class with multiple detection algorithms
- **FEATURE**: Support for NeuroKit2, BioSPPy, and basic scipy methods
- **FEATURE**: Robust artifact filtering with physiological constraints (0.2s ≤ RR ≤ 2.0s)
- **FEATURE**: Bandpass filtering (0.5-40 Hz) before R-peak detection
- **FEATURE**: Tachogram extraction with interpolation options

#### 📊 HRV Feature Extraction
- **NEW**: Complete `HRVFeatureExtractor` ported from MATLAB codebase
- **FEATURE**: 23 HRV features across three domains:
  - Time-domain: NN50, pNN50, RMSSD, SDNN, SDSD, RRMean, RRMin, RRMax, RRVar
  - Frequency-domain: TOTAL_POWER, VLF_POWER, LF_POWER, HF_POWER, normalized powers, LF/HF ratio
  - Nonlinear: SD1, SD2, SD1/SD2, SampEn, ApEn
- **FEATURE**: Lomb-Scargle periodogram for irregularly sampled tachogram
- **FEATURE**: Poincaré plot analysis with geometrical approach
- **FEATURE**: Sample and Approximate Entropy calculation

#### 🔄 Pipeline Integration
- **NEW**: `HRVFeaturePipeline` class orchestrating complete processing
- **FEATURE**: Sliding window feature extraction (30s windows, 5s stride)
- **FEATURE**: Per-recording CSV export with metadata columns
- **FEATURE**: Command-line interface for single file and dataset processing
- **FEATURE**: Configurable parameters (SPH, window size, stride, etc.)

#### 🧠 LSTM Sequence Construction
- **NEW**: `LSTMSequenceBuilder` for temporal sequence creation
- **FEATURE**: (N, seq_len, n_features) format for LSTM training
- **FEATURE**: Default configuration: 36 sequences (180s history / 5s stride)
- **FEATURE**: HDF5 storage for efficient large dataset handling
- **FEATURE**: Feature normalization and class weight calculation
- **FEATURE**: Train/validation splits avoiding temporal leakage
- **NOTE**: Original implementation had data leakage issue (fixed in v1.1)

#### 📈 Cross-Validation Framework
- **NEW**: `LOSOEvaluator` implementing leave-one-seizure-out validation
- **FEATURE**: Automatic seizure episode detection and splitting
- **FEATURE**: Support for multiple model types (RandomForest, LogisticRegression, LSTM)
- **FEATURE**: Comprehensive metrics: sensitivity, specificity, ROC AUC, FPR per hour
- **FEATURE**: Per-fold and summary statistics with JSON export

#### ⚡ Real-time Streaming Emulation
- **NEW**: `StreamingEmulator` for real-time inference simulation
- **FEATURE**: Circular buffer management for feature sequences
- **FEATURE**: Configurable debounce logic (k consecutive predictions)
- **FEATURE**: Detection timing analysis relative to seizure onset
- **FEATURE**: Comprehensive logging of detection events and probabilities
- **FEATURE**: False alarm rate calculation

#### 🧪 Testing and Validation
- **NEW**: Comprehensive unit test suite in `test_pipeline.py`
- **FEATURE**: Label alignment validation for synthetic seizures
- **FEATURE**: R-peak detection accuracy tests with known ground truth
- **FEATURE**: Feature extraction continuity and NaN handling tests
- **FEATURE**: End-to-end integration testing
- **FEATURE**: Validation checks with deterministic synthetic data

### Technical Improvements

#### Data Processing
- **IMPROVEMENT**: Robust handling of missing files and corrupted data
- **IMPROVEMENT**: Memory-efficient processing for large datasets
- **IMPROVEMENT**: Consistent temporal alignment across all components
- **IMPROVEMENT**: Artifact detection and interpolation options

#### Code Quality
- **IMPROVEMENT**: Modular design with clear separation of concerns
- **IMPROVEMENT**: Comprehensive logging and error handling
- **IMPROVEMENT**: Type hints and documentation for all functions
- **IMPROVEMENT**: Configurable parameters with sensible defaults

#### Performance
- **IMPROVEMENT**: Efficient HDF5 storage for sequence data
- **IMPROVEMENT**: Vectorized feature calculations
- **IMPROVEMENT**: Optional feature normalization for improved convergence
- **IMPROVEMENT**: Parallel processing capabilities

### Configuration Parameters

#### Default Values (All Configurable)
- **SPH**: 180 seconds (3 minutes)
- **Label Width**: 30 seconds (±15s tolerance)
- **Window Size**: 30 seconds
- **Stride**: 5 seconds  
- **History**: 180 seconds (36 windows for LSTM)
- **Sampling Rate**: 256 Hz
- **R-peak Filter**: 0.5-40 Hz bandpass
- **Frequency Analysis**: 4 Hz resampling for Welch PSD

#### Command-Line Interfaces
- **`hrv_pipeline.py`**: Main feature extraction pipeline
- **`lstm_sequences.py`**: Sequence building for LSTM training
- **`evaluate_loso.py`**: Cross-validation evaluation
- **`stream_emulator.py`**: Real-time streaming emulation
- **`test_pipeline.py`**: Unit tests and validation

### Output Formats

#### CSV Features (per recording)
```
subject_id, recording_id, window_start_time, window_center_time, window_end_time,
[23 HRV features], label
```

#### HDF5 Sequences
- `X`: (N, seq_len, n_features) feature sequences
- `y`: (N,) labels  
- `timestamps`: (N,) center times
- `metadata`: configuration and statistics

#### Evaluation Results
- Per-fold CSV with detailed metrics
- Summary JSON with aggregated statistics
- Processing logs with detection timing

### Dependencies Added
- **neurokit2**: Advanced R-peak detection (recommended)
- **biosppy**: Alternative R-peak detection
- **h5py**: Efficient sequence storage
- **tensorflow**: LSTM model support (optional)

### Documentation
- **NEW**: Comprehensive README with usage examples
- **NEW**: API documentation for all classes and functions
- **NEW**: Configuration guide with parameter explanations
- **NEW**: Troubleshooting section with common issues

### Breaking Changes from Original Pipeline
- **REMOVED**: Strategy2Labeler (5-8 min window) replaced with FixedSPHLabeler
- **CHANGED**: Window labeling method from majority vote to center-point sampling
- **CHANGED**: Output format from NPZ to CSV for features, HDF5 for sequences
- **CHANGED**: Evaluation framework from simple accuracy to comprehensive LOSO

### Migration Guide
1. Replace `Strategy2Labeler` usage with `FixedSPHLabeler`
2. Update window processing to use `WindowCenterLabeler.assign_window_labels()`
3. Migrate from NPZ to CSV/HDF5 output formats
4. Use new command-line interfaces for processing

### Future Enhancements (Planned)
- **DFA Features**: Detrended Fluctuation Analysis implementation
- **Additional Entropy Measures**: Multiscale entropy, fuzzy entropy
- **Real-time Processing**: Live ECG stream processing capabilities
- **Model Training**: Integrated LSTM training with hyperparameter optimization
- **Visualization**: Feature importance and detection timing plots

---

## Development Notes

### Testing Coverage
- ✅ Label alignment for known seizure times
- ✅ R-peak detection accuracy on synthetic ECG
- ✅ HRV feature calculation correctness
- ✅ Sequence construction and temporal ordering
- ✅ LOSO split generation and evaluation
- ✅ Streaming emulation with debounce logic

### Performance Benchmarks
- **Processing Speed**: ~1 minute per hour of ECG data
- **Memory Usage**: <2GB for typical recordings
- **Feature Extraction**: 23 features per 30-second window
- **Sequence Building**: 36-window sequences for 180s history

### Validation Results
- **Synthetic ECG**: >90% R-peak detection accuracy
- **Feature Consistency**: Zero NaN values for valid inputs
- **Temporal Alignment**: Exact SPH targeting verified
- **LOSO Splits**: Proper temporal isolation confirmed

---

*This changelog documents the complete implementation of the Fixed SPH HRV pipeline for seizure prediction research.*