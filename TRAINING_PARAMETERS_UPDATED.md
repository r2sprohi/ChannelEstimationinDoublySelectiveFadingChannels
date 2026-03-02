# Updated Training Parameters

## Summary of Changes

The following parameters have been updated to increase the dataset size and training intensity:

### 1. Dataset Size
- **Previous**: 100 total samples (80 training + 20 testing)
- **Updated**: 10,000 total samples (8,000 training + 2,000 testing)
- **Files Modified**:
  - `IDX_Generation.m`: Changed `IDX = 100` to `IDX = 10000`
  - `main.m`: Changed `load('./samples_indices_100.mat')` to `load('./samples_indices_10000.mat')`
  - `DNN_Datasets_Generation.m`: Updated to load `samples_indices_10000.mat`
  - `DNN_Results_Processing_Fixed.m`: Updated to load `samples_indices_10000.mat`

### 2. Training Parameters
- **Batch Size**: 128 (previously 32)
- **Epochs**: 500 (previously 100)

These parameters are passed as command-line arguments to `DNN.py` during training.

## Steps to Use Updated Configuration

### Step 1: Generate New Sample Indices
```matlab
% Run in MATLAB
run IDX_Generation.m
```
This will create `samples_indices_10000.mat` with 8,000 training and 2,000 testing indices.

### Step 2: Generate Training Data
```matlab
% In main.m, set:
configuration = 'training';
EbN0dB = 40;
% Run main.m (will take ~8000x longer than before!)
```

### Step 3: Generate Training Dataset for DNN
```matlab
% In DNN_Datasets_Generation.m, set:
configuration = 'training';
% Run DNN_Datasets_Generation.m
```

### Step 4: Train DNN Model (Updated Command)
```bash
python DNN.py High VTV_SDWW QPSK 40 312 104 256 128 64 500 128
#             mob  channel   mod  snr in  out h1  h2  h3 ep  bs
```

**Training Details**:
- Total training samples: 8,000
- Validation split: 25% (2,000 samples for validation)
- Actual training samples: 6,000
- Batch size: 128 (47 batches per epoch)
- Epochs: 500 (with early stopping)
- Expected training time: Significantly longer (estimated several hours depending on hardware)

### Step 5: Generate Testing Data
```matlab
% In main.m, set:
configuration = 'testing';
EbN0dB = 0:5:40;
% Run main.m for all SNR values
```

### Step 6: Generate Testing Datasets for DNN
```matlab
% In DNN_Datasets_Generation.m, set:
configuration = 'testing';
% Run DNN_Datasets_Generation.m
```

### Step 7: Test DNN Model
```bash
python DNN.py High VTV_SDWW QPSK 40
```

### Step 8: Process and Plot Results
```matlab
% Run in MATLAB
run DNN_Results_Processing_Fixed.m
```

## Expected Improvements

With 80x more training data (8,000 vs 100 samples):
1. **Better generalization** across different channel realizations
2. **Improved performance** at all SNR levels
3. **More robust** to channel variations
4. **Smoother learning curves** due to larger batch size
5. **Better convergence** with more epochs

## Important Notes

1. **Training time**: Generating 10,000 channel realizations in `main.m` will take approximately **80-100x longer** than before (~several hours depending on your system)

2. **Memory**: Ensure you have sufficient RAM and disk space:
   - Each simulation file: ~50-100 MB
   - Total storage needed: ~5-10 GB

3. **Early stopping**: The training uses early stopping with patience=20, so it may stop before 500 epochs if validation loss plateaus

4. **Learning rate reduction**: Learning rate reduces by 10x if validation loss doesn't improve for 15 epochs

5. **Files to keep**:
   - Keep the old `samples_indices_100.mat` for reference
   - New file will be `samples_indices_10000.mat`

## Verification

After running `IDX_Generation.m`, verify:
```matlab
load('./samples_indices_10000.mat');
fprintf('Training samples: %d\n', length(training_samples));  % Should be 8000
fprintf('Testing samples: %d\n', length(testing_samples));    % Should be 2000
```

## Computational Requirements

- **MATLAB simulation**: ~10-20 hours (depending on CPU)
- **Python training**: ~2-6 hours (depending on GPU/CPU)
- **Total workflow**: Plan for ~1-2 days of computation time
