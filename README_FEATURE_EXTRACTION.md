# Fixed SPH HRV Pipeline for Seizure Prediction

This repository implements a complete pipeline for extracting HRV (Heart Rate Variability) features from ECG data with Fixed SPH (Seizure Prediction Horizon) labeling for seizure prediction exactly 180 seconds before seizure onset.

## 🚀 Quick Start

**For Local Files:**
```bash
python data_processing_pipeline.py
```

**For AWS S3 Files:**
```bash
# Configure AWS credentials first (see AWS S3 Setup below)
python data_processing_pipeline.py
```

This processes your entire dataset with optimized settings and outputs CSV files ready for LSTM training.

## ☁️ **AWS S3 Support**

The pipeline now supports both local files and AWS S3 storage. You can:
- **Input:** Read EEG/ECG files directly from S3
- **Output:** Save CSV feature files directly to S3
- **Mixed:** Use local input with S3 output, or vice versa

### **S3 Configuration in `data_processing_pipeline.py`:**

```python
# LOCAL FILES (comment/uncomment as needed)
#data_root = "/Volumes/Seizury/ds005873"
#output_dir = "/Volumes/Seizury/HRV/hrv_features"

# AWS S3 (comment/uncomment as needed)
data_root = "s3://seizury-data/ds005873"
output_dir = "s3://seizury-data/hrv_features"
```

### **AWS Setup Requirements:**

1. **Install boto3:**
   ```bash
   pip install boto3
   ```

2. **Configure AWS credentials** (choose one method):
   
   **Method 1: AWS CLI**
   ```bash
   pip install awscli
   aws configure
   ```
   
   **Method 2: Environment variables**
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=us-east-1
   ```
   
   **Method 3: AWS credentials file** (`~/.aws/credentials`)
   ```ini
   [default]
   aws_access_key_id = your_access_key
   aws_secret_access_key = your_secret_key
   region = us-east-1
   ```

3. **Verify S3 access:**
   ```bash
   aws s3 ls s3://seizury-data/
   ```

## 🎯 Which Pipeline to Use?

### **Option 1: Simple Batch Processing** ⭐ **RECOMMENDED**
**File:** `data_processing_pipeline.py`
- **Best for:** Processing entire dataset quickly
- **Configuration:** Pre-configured for your dataset paths
- **Usage:** Just run `python data_processing_pipeline.py`
- **Output:** CSV files in `/Volumes/Seizury/HRV/hrv_features/`

### **Option 2: Advanced Configuration**
**File:** `hrv_pipeline.py`  
- **Best for:** Custom parameters, single files, research experiments
- **Configuration:** Full command-line interface
- **Usage:** See detailed examples below
- **Output:** Configurable output directory

**Both pipelines produce identical results with Fixed SPH labeling!**

## Overview

The pipeline transforms ECG signals into HRV feature sequences suitable for LSTM training, targeting seizure prediction at a fixed 3-minute warning horizon with a narrow tolerance window.

## 📁 **Pipeline Files Explained**

### **Main Processing Files:**
1. **`data_processing_pipeline.py`** ⭐ **START HERE**
   - **Purpose:** Batch process entire dataset with optimal settings
   - **When to use:** When you want to process the full dataset quickly
   - **Configuration:** Fixed paths optimized for your dataset
   - **Command:** `python data_processing_pipeline.py`
   - **Output:** CSV files in `/Volumes/Seizury/HRV/hrv_features/`

2. **`hrv_pipeline.py`** 
   - **Purpose:** Advanced pipeline with full configuration options
   - **When to use:** Custom parameters, single files, research experiments
   - **Configuration:** Full command-line interface with many options
   - **Command:** See usage examples below

### **Supporting Modules:**
3. **`labelers.py`** - Fixed SPH labeling (180s ± 15s before seizure)
4. **`ecg_processing.py`** - R-peak detection and tachogram extraction  
5. **`hrv_features.py`** - 23 HRV feature calculations
6. **`lstm_sequences.py`** - Create LSTM-ready sequences with patient-level splits
7. **`evaluate_loso.py`** - Leave-one-seizure-out cross-validation
8. **`stream_emulator.py`** - Real-time streaming simulation

## 🔧 **Configuration Parameters** (Used by Both Pipelines)

### Default Values (All Configurable)
- **SPH**: 180 seconds (3 minutes)
- **Label Width**: 30 seconds (±15s tolerance)
- **Window Size**: 30 seconds
- **Stride**: 5 seconds  
- **History**: 180 seconds (36 windows for LSTM)
- **Sampling Rate**: 256 Hz
- **R-peak Filter**: 0.5-40 Hz bandpass
- **Frequency Analysis**: 4 Hz resampling for Welch PSD

## 🎯 **Usage Examples**

### **1. Quick Start - Process Entire Dataset** ⭐ **RECOMMENDED**

```bash
# Simple - just run this for your dataset:
python data_processing_pipeline.py

# Output: CSV files in /Volumes/Seizury/HRV/hrv_features/
# ✅ Fixed SPH labeling (180s ± 15s)
# ✅ 23 HRV features extracted
# ✅ Ready for LSTM training
```

### **2. Advanced Usage - Custom Configuration**

**Process entire dataset with custom settings:**
```bash
python hrv_pipeline.py \
    --data-root /Volumes/Seizury/ds005873 \
    --output-dir ./custom_features \
    --sph 180 \
    --label-width 30
```

**Process single recording:**
```bash
python hrv_pipeline.py \
    --single-file /path/to/ecg_file.edf \
    --eeg-file /path/to/eeg_file.edf \
    --annotation-file /path/to/annotations.tsv \
    --output-dir ./features
```

### **3. Create LSTM Sequences** ⚠️ **CRITICAL: Use Patient-Level Splits**

```bash
# ✅ CORRECT: Patient-level splits (prevents data leakage)
python lstm_sequences.py \
    --input-dir /Volumes/Seizury/HRV/hrv_features \
    --output-dir ./sequences \
    --seq-len 36 \
    --history 180 \
    --normalize \
    --create-splits
```

**Alternative: Specify exact patients for test/validation**
```bash
python lstm_sequences.py \
    --input-dir /Volumes/Seizury/HRV/hrv_features \
    --output-dir ./sequences \
    --create-splits \
    --test-subjects sub-01 sub-02 sub-05 \
    --val-subjects sub-03 sub-04 sub-08 \
    --normalize
```

### **4. Run Evaluation**

```bash
# Leave-one-seizure-out cross-validation
python evaluate_loso.py \
    --input-dir /Volumes/Seizury/HRV/hrv_features \
    --output-dir ./evaluation_results \
    --model rf

# Streaming emulation
python stream_emulator.py \
    --model ./trained_model.pkl \
    --ecg-file /path/to/test_ecg.edf \
    --annotation-file /path/to/test_annotations.tsv \
    --threshold 0.5 \
    --debounce-k 3
```

## 📋 **Complete Workflow**

```bash
# Step 1: Extract HRV features (choose one)
python3 data_processing_pipeline.py                    # Simple batch processing
# OR
python3 hrv_pipeline.py --data-root /your/dataset      # Advanced configuration

# Step 2: Create LSTM sequences with patient-level splits
python3 lstm_sequences.py --input-dir hrv_features --create-splits --normalize

# Step 3: Train your model (your code)
# Use: train_sequences.h5, val_sequences.h5, test_sequences.h5

# Step 4: Evaluate with LOSO cross-validation
python3 evaluate_loso.py --input-dir hrv_features --model your_model

# Step 5: Test real-time performance
python3 stream_emulator.py --model your_model.pkl --ecg-file test.edf
```

## 📊 **Output Formats** 

### **Feature CSV Format** (Main Output)

Each recording produces a CSV with the following columns:

```
subject_id,recording_id,window_start_time,window_center_time,window_end_time,
NN50,pNN50,RMSSD,SDNN,SDSD,RRMean,RRMin,RRMax,RRVar,
TOTAL_POWER,VLF_POWER,LF_POWER,HF_POWER,VLF_NORM,LF_NORM,HF_NORM,LF_TO_HF,
SD1,SD2,SD1toSD2,SampEn,ApEn,
label
```

**Example output structure:**
```
/Volumes/Seizury/HRV/hrv_features/
├── sub-01_ses-01_run-01_features.csv
├── sub-01_ses-01_run-02_features.csv
├── sub-02_ses-01_run-01_features.csv
├── ...
├── combined_features.csv          # All recordings combined
├── processing_summary.csv         # Processing statistics
└── dataset_info.json             # Dataset metadata
```

### **LSTM Sequence Format** (HDF5)

When using `lstm_sequences.py`:

**Main sequences file:**
- `sequences.h5`: Complete dataset with all patients
  - `X`: (N, seq_len, n_features) - Feature sequences
  - `y`: (N,) - Labels for each sequence
  - `timestamps`: (N,) - Center timestamps
  - `subjects`: (N,) - Subject IDs
  - `recordings`: (N,) - Recording IDs

**Patient-level splits** (when using --create-splits):
- `train_sequences.h5`: Training patients only
- `val_sequences.h5`: Validation patients only  
- `test_sequences.h5`: Test patients only
- `patient_splits.json`: Split information and patient assignments

**Example patient_splits.json:**
```json
{
  "train_subjects": ["sub-02", "sub-04", "sub-06", ...],
  "val_subjects": ["sub-03", "sub-08", "sub-15", "sub-20"],
  "test_subjects": ["sub-01", "sub-05", "sub-12", "sub-18", "sub-22"],
  "train_sequences": 15420,
  "val_sequences": 3105,
  "test_sequences": 3890
}
```

## 🔬 **HRV Features Implemented** (23 Total)

### Time-Domain (9 features)
- **NN50**: Number of successive RR intervals differing by >50ms
- **pNN50**: Percentage of NN50
- **RMSSD**: Root mean square of successive differences
- **SDNN**: Standard deviation of RR intervals
- **SDSD**: Standard deviation of successive differences
- **RRMean**: Mean RR interval
- **RRMin**: Minimum RR interval
- **RRMax**: Maximum RR interval
- **RRVar**: Variance of RR intervals

### Frequency-Domain (8 features)
- **TOTAL_POWER**: Total power in all frequency bands
- **VLF_POWER**: Very low frequency power (0.003-0.04 Hz)
- **LF_POWER**: Low frequency power (0.04-0.15 Hz)
- **HF_POWER**: High frequency power (0.15-0.4 Hz)
- **VLF_NORM**: Normalized VLF power
- **LF_NORM**: Normalized LF power
- **HF_NORM**: Normalized HF power
- **LF_TO_HF**: LF/HF ratio

### Nonlinear (6 features)
- **SD1**: Poincaré plot short-term variability
- **SD2**: Poincaré plot long-term variability
- **SD1toSD2**: SD1/SD2 ratio
- **SampEn**: Sample entropy
- **ApEn**: Approximate entropy

*Note: DFA features are planned for future implementation*

## 🧪 **Installation & Dependencies**

### **Requirements**

```bash
pip install numpy pandas scipy scikit-learn mne h5py
pip install neurokit2  # Recommended for R-peak detection
pip install biosppy    # Alternative R-peak detection
pip install tensorflow # For LSTM models (optional)
pip install boto3      # For AWS S3 support (optional)
```

### **Setup**

1. Clone or download the repository
2. Install dependencies: `pip install -r requirements.txt`
3. **For Local Files:** Ensure dataset is available at `/Volumes/Seizury/ds005873`
4. **For AWS S3:** Configure AWS credentials (see AWS S3 Setup above)
5. Edit `data_processing_pipeline.py` to set your data source (local or S3)
6. Run: `python data_processing_pipeline.py`

## ⚠️ **Critical: Preventing Data Leakage**

**ALWAYS use patient-level splits for seizure prediction!**

### **The Problem**
- **Wrong**: Random splitting mixes patients between train/test sets
- **Result**: Model learns patient-specific patterns, not seizure patterns
- **Consequence**: Inflated performance that doesn't generalize to new patients

### **The Solution**
- **Correct**: Split by patient before any training
- **Implementation**: Use `--create-splits` flag in `lstm_sequences.py`
- **Verification**: Check `patient_splits.json` to ensure no patient overlap

### **Example of Proper Splitting**
```bash
# ✅ CORRECT: Patient-level splits
python lstm_sequences.py --input-dir hrv_features --create-splits

# ❌ WRONG: Would create random splits if --create-splits not used
python lstm_sequences.py --input-dir hrv_features  # Don't do this for final training!
```

### **Validation**
The pipeline automatically logs patient assignments:
```
INFO: Test patients (5): ['sub-01', 'sub-05', 'sub-12', 'sub-18', 'sub-22']
INFO: Val patients (4): ['sub-03', 'sub-08', 'sub-15', 'sub-20']  
INFO: Train patients (16): ['sub-02', 'sub-04', ...]
```

**Verify that no patient appears in multiple sets!**

## 🔧 **Key Implementation Notes**

### **Fixed SPH Labeling**
- **Target**: seizure_onset - 180 seconds
- **Tolerance**: ±15 seconds window
- **Labels**: 0=Normal, 1=Pre-seizure, 2=Seizure

### **Temporal Alignment**
- All times are referenced to window centers for consistent alignment
- SPH target calculated as: `t_target = seizure_onset - 180s`
- Label window: `[t_target - 15s, t_target + 15s]`

### **R-peak Detection Priority**
1. NeuroKit2 (if available) - most robust
2. BioSPPy (if available) - good alternative
3. Basic scipy.signal method - fallback

### **Artifact Handling**
- RR interval filtering: 0.2s ≤ RR ≤ 2.0s (physiological range)
- Missing features replaced with zeros
- Optionally interpolate artifacts in tachogram

## 🏁 **File Structure Summary**

```
├── data_processing_pipeline.py     # ⭐ MAIN: Simple batch processing
├── hrv_pipeline.py                 # Advanced pipeline with CLI options
├── labelers.py                     # Fixed SPH labeling (180s ± 15s)
├── ecg_processing.py               # R-peak detection and tachogram
├── hrv_features.py                 # 23 HRV feature calculations
├── lstm_sequences.py               # Sequence building for LSTM
├── evaluate_loso.py                # Cross-validation evaluation
├── stream_emulator.py              # Real-time inference emulation
├── test_pipeline.py                # Unit tests and validation
├── README_FEATURE_EXTRACTION.md    # This documentation
└── CHANGELOG.md                    # Version history and changes
```

## 🎯 **Expected Performance**

Based on similar HRV-based seizure prediction studies:
- **Sensitivity**: 60-80% per seizure
- **False Positive Rate**: 0.1-1.0 per hour
- **Warning Time**: 180 ± 30 seconds (by design)

## 🧪 **Testing & Validation**

### **Run Unit Tests**
```bash
python test_pipeline.py
```

### **Key Validation Tests**
1. **Label Alignment**: Verifies SPH target at exactly 180s before seizure onset
2. **R-peak Detection**: Tests accuracy on synthetic ECG with known R-peak times
3. **Feature Continuity**: Ensures no NaN values in feature extraction
4. **Sequence Construction**: Validates temporal ordering and shape consistency

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Missing Dependencies**
   ```bash
   pip install neurokit2 biosppy  # For R-peak detection
   ```

2. **Memory Issues with Large Datasets**
   - Process recordings individually
   - Use stride > 1 for sequence building
   - Consider feature normalization

3. **Poor R-peak Detection**
   - Check ECG signal quality
   - Adjust bandpass filter parameters
   - Try different detection methods

4. **No Seizures Found**
   - Verify annotation file format
   - Check `eventType` column naming
   - Ensure seizure events are properly tagged

### **Performance Optimization**
- Use `--normalize` flag for LSTM sequences
- Process in parallel using multiple cores
- Store intermediate results to avoid reprocessing

## 📖 **Citation**

If you use this pipeline in your research, please cite:

```
ECG-based Seizure Prediction Pipeline with Fixed SPH Labeling
Version 1.1, 2024
```

## 🤝 **Contributing**

1. Follow the existing code style and documentation patterns
2. Add unit tests for new features
3. Update this README for any new functionality
4. Ensure backward compatibility with existing data formats

## 📄 **License**

This project is part of the seizure prediction research framework. Please contact the authors for usage permissions and collaboration opportunities.