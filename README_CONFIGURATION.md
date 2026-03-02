# Channel Estimation DNN Configuration Guide

This document describes the critical configuration parameters that must remain consistent across all files in the project.

## Critical Parameters

### 1. Number of Active Subcarriers (nUSC)
- **Value**: 52
- **Calculation**: nDSC (48) + nPSC (4) = 52
- **Files**:
  - `main.m` line 11: `nUSC = nDSC + nPSC`
  - `DNN_Datasets_Generation.m` line 9: `nUSC = 52`
  - `DNN.py` line 21: Comment indicates `nUSC = 52`

**DNN Implications**:
- Input dimension: `6 * nUSC = 312`
- Output dimension: `2 * nUSC = 104`

### 2. Number of OFDM Symbols per Realization (nSym)
- **Value**: 50
- **Files**:
  - `main.m` line 16: `nSym = 50`
  - `DNN_Datasets_Generation.m` line 10: `nSym = 50`

### 3. Channel Model (ChType / channel_model)
- **Value**: 'VTV_SDWW'
- **Files**:
  - `main.m` line 60: `ChType = 'VTV_SDWW'`
  - `DNN_Datasets_Generation.m` line 13: `ChType = 'VTV_SDWW'`
  - `DNN.py`: Pass as `channel_model` argument

### 4. Modulation Scheme (modu / modulation)
- **Value**: 'QPSK'
- **Files**:
  - `main.m` line 44: `modu = 'QPSK'`
  - `DNN_Datasets_Generation.m` line 12: `modu = 'QPSK'`
  - `DNN.py`: Pass as `modulation` argument

### 5. Mobility Configuration
- **Value**: 'High'
- **Files**:
  - `DNN_Datasets_Generation.m` line 11: `mobility = 'High'`
  - `DNN.py`: Pass as `mobility` argument

### 6. Configuration Mode
- **Values**: 'training' or 'testing'
- **Files**:
  - `main.m` line 69: `configuration = 'training'` (or 'testing')
  - `DNN_Datasets_Generation.m` line 5: `configuration = 'training'` (or 'testing')

### 7. SNR Values

#### Training SNR
- **Value**: 40 dB
- **Files**:
  - `main.m` line 71: `EbN0dB = 40` (for training)
  - `DNN_Datasets_Generation.m` line 25: `EbN0dB = 40` (for training)
  - `DNN.py`: Pass as `training_snr` argument

#### Testing SNR Range
- **Values**: 0:5:40 (i.e., [0, 5, 10, 15, 20, 25, 30, 35, 40] dB)
- **Files**:
  - `DNN_Datasets_Generation.m` line 27: `EbN0dB = 0:5:40` (for testing)
  - `DNN.py` line 19: `SNR_INDEX = np.arange(0, 45, 5)`

## File Naming Conventions

### Simulation Files (MATLAB)
Format: `{mobility}_{ChType}_{modu}_{configuration}_simulation_{SNR}.mat`

Example:
- `High_VTV_SDWW_QPSK_training_simulation_40.mat`
- `High_VTV_SDWW_QPSK_testing_simulation_0.mat`

### Dataset Files (MATLAB → Python)
Format: `{mobility}_{ChType}_{modu}_DNN_{configuration}_dataset_{SNR}.mat`

Example:
- `High_VTV_SDWW_QPSK_DNN_training_dataset_40.mat`
- `High_VTV_SDWW_QPSK_DNN_testing_dataset_0.mat`

### Model Files (Python)
Format: `{mobility}_{channel_model}_{modulation}_DNN_{training_snr}.h5`

Example:
- `High_VTV_SDWW_QPSK_DNN_40.h5`

### Scaler Files (Python)
Format: `scaler_{x/y}_{training_snr}.pkl`

Example:
- `scaler_x_40.pkl`
- `scaler_y_40.pkl`

### Results Files (Python)
Format: `{mobility}_{channel_model}_{modulation}_DNN_results_{SNR}.mat`

Example:
- `High_VTV_SDWW_QPSK_DNN_results_0.mat`
- `High_VTV_SDWW_QPSK_DNN_results_5.mat`

## Workflow

### 1. Generate Simulation Data (MATLAB)
```matlab
% In main.m, set:
configuration = 'training';
EbN0dB = 40;
% Run main.m
```

### 2. Generate DNN Dataset (MATLAB)
```matlab
% In DNN_Datasets_Generation.m, set:
configuration = 'training';
% Run DNN_Datasets_Generation.m
```

### 3. Train DNN Model (Python)
```bash
python3 DNN.py High VTV_SDWW QPSK 40 312 104 256 128 64 500 128
#             mob  channel   mod  snr in  out h1  h2  h3 ep  bs
```

Parameters:
- mobility: High
- channel_model: VTV_SDWW
- modulation: QPSK
- training_snr: 40
- input_dim: 312 (= 6 * 52)
- output_dim: 104 (= 2 * 52)
- h1, h2, h3: Hidden layer sizes (256, 128, 64)
- epochs: 500 (increased from 100)
- batch_size: 128 (increased from 32)

**Note**: With 8000 training samples (80% of 10000), batch_size=128, and validation_split=0.25:
- Training samples: 6000 (75% of 8000)
- Validation samples: 2000 (25% of 8000)
- Testing samples: 2000 (from separate test set)

### 4. Generate Testing Data (MATLAB)
```matlab
% In main.m, set:
configuration = 'testing';
EbN0dB = 0:5:40;
% Run main.m for each SNR

% In DNN_Datasets_Generation.m, set:
configuration = 'testing';
% Run DNN_Datasets_Generation.m
```

### 5. Test DNN Model (Python)
```bash
python DNN.py High VTV_SDWW QPSK 40
#             mob  channel   mod  snr
```

The script will automatically test over all SNR values in `SNR_INDEX`.

## Common Issues

### Issue 1: Dimension Mismatch
**Error**: `Input dimension mismatch! Expected 312, got XXX`

**Solution**: Verify that `nUSC = 52` in both `main.m` and `DNN_Datasets_Generation.m`

### Issue 2: File Not Found
**Error**: `Dataset file not found: ...`

**Solution**:
1. Ensure you ran `main.m` with the correct `configuration` setting
2. Ensure you ran `DNN_Datasets_Generation.m` after `main.m`
3. Check that file naming matches the expected format

### Issue 3: Scaler Not Found
**Error**: `Scaler not found: scaler_x_40.pkl`

**Solution**: Run training mode first to generate the scalers before testing

### Issue 4: SNR Mismatch
**Error**: `WARNING: Dataset not found for SNR=X dB, skipping...`

**Solution**: Generate testing datasets for all SNR values (0:5:40) using `main.m` and `DNN_Datasets_Generation.m`

## Validation Checklist

Before running the complete workflow, verify:

- [ ] `nUSC = 52` in all files
- [ ] `nSym = 50` in MATLAB files
- [ ] Channel model matches: 'VTV_SDWW'
- [ ] Modulation matches: 'QPSK'
- [ ] Mobility matches: 'High'
- [ ] Configuration mode is set correctly ('training' or 'testing')
- [ ] SNR values are correct (40 for training, 0:5:40 for testing)
- [ ] File naming conventions are followed
- [ ] All required files exist before running each step
