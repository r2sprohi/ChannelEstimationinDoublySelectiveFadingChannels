# Visualization Guide: MSE and BER Comparison

This guide explains how to visualize the performance comparison between DNN, MMSE-VP, and LS channel estimators.

## Summary of Changes

### Files Modified

1. **[main.m](main.m)** - Updated to save additional data:
   - `MMSE_VP_Estimate_Structure` - MMSE-VP channel estimates
   - `TX_Bits_Stream_Structure` - Transmitted bits for BER calculation
   - `Random_permutation_Vector` - Interleaver permutation for decoding

2. **[DNN_Results_Processing_Fixed.m](DNN_Results_Processing_Fixed.m)** - NEW FILE:
   - Fixed version of your results processing script
   - Loads DNN, LS, and MMSE results
   - Calculates MSE and BER for all methods
   - Generates comparison plots

## Inconsistencies Fixed in Original File

### 1. **File Naming Mismatch**
❌ **Original**: `{mobility}_{ChType}_{modu}_{scheme}_DNN_Results_{SNR}.mat`
✅ **Fixed**: `{mobility}_{ChType}_{modu}_DNN_results_{SNR}.mat`

### 2. **Variable Names Mismatch**
❌ **Original**: Expected `STA_DNN_test_y_40`, `STA_DNN_corrected_y_40`
✅ **Fixed**: Uses `DNN_input_X`, `DNN_true_Y`, `DNN_predicted_Y` (from DNN.py)

### 3. **Missing MMSE and LS Data**
❌ **Original**: Only loaded DNN results
✅ **Fixed**: Loads LS and MMSE estimates from simulation files

### 4. **Missing Simulation Parameters**
❌ **Original**: Tried to load non-existent `simulation_parameters.mat`
✅ **Fixed**: Removed dependency, uses hardcoded parameters consistent with main.m

### 5. **Incorrect Data Reshaping**
❌ **Original**: Assumed different data format
✅ **Fixed**: Correctly reshapes `[N_samples, 2*nUSC]` → `[nUSC, nSym, N_ch]`

### 6. **Missing TX Bits**
❌ **Original**: Expected `TX_Bits_Stream_Structure` but wasn't saving it
✅ **Fixed**: main.m now saves transmitted bits for BER calculation

## Complete Workflow for Visualization

### Step 1: Generate Testing Simulation Data (MATLAB)

Update [main.m](main.m#L66-L68):
```matlab
configuration = 'testing';
indices = testing_samples;
EbN0dB = 0:5:40;  % Test over all SNR values
```

Run for each SNR:
```matlab
% In MATLAB
main
```

**Output files**:
- `High_VTV_SDWW_QPSK_testing_simulation_0.mat`
- `High_VTV_SDWW_QPSK_testing_simulation_5.mat`
- ... (all SNR values from 0 to 40 in steps of 5)

Each file now contains:
- `Received_Symbols_FFT_Structure`
- `True_Channels_Structure`
- `LS_Estimate_Structure`
- `Prev_Channel_Estimate_Structure`
- `MMSE_VP_Estimate_Structure` ← NEW
- `TX_Bits_Stream_Structure` ← NEW
- `Random_permutation_Vector` ← NEW

### Step 2: Generate Testing DNN Datasets (MATLAB)

Update [DNN_Datasets_Generation.m](DNN_Datasets_Generation.m#L5):
```matlab
configuration = 'testing';
```

Run:
```matlab
% In MATLAB
DNN_Datasets_Generation
```

**Output files**:
- `High_VTV_SDWW_QPSK_DNN_testing_dataset_0.mat`
- `High_VTV_SDWW_QPSK_DNN_testing_dataset_5.mat`
- ... (all SNR values)

### Step 3: Run DNN Testing (Python)

```bash
python DNN.py High VTV_SDWW QPSK 40
```

**Output files**:
- `High_VTV_SDWW_QPSK_DNN_results_0.mat`
- `High_VTV_SDWW_QPSK_DNN_results_5.mat`
- ... (all SNR values)

Each file contains:
- `DNN_input_X` - Input features
- `DNN_true_Y` - True channel (real + imaginary)
- `DNN_predicted_Y` - DNN predicted channel (real + imaginary)

### Step 4: Process and Visualize Results (MATLAB)

Run the fixed processing script:
```matlab
% In MATLAB
DNN_Results_Processing_Fixed
```

**Output**:
1. **Figures**:
   - `High_VTV_SDWW_QPSK_MSE_comparison.png` - MSE vs SNR plot
   - `High_VTV_SDWW_QPSK_BER_comparison.png` - BER vs SNR plot
   - `.fig` versions for editing in MATLAB

2. **Data**:
   - `High_VTV_SDWW_QPSK_comparison_results.mat` - Numerical results

## What the Plots Show

### MSE vs SNR Plot
- **X-axis**: SNR in dB (0 to 40)
- **Y-axis**: Normalized Mean Squared Error (log scale)
- **Lines**:
  - Blue (circles): LS estimator
  - Red (squares): DNN estimator
  - Green (triangles): MMSE-VP estimator

**Expected behavior**:
- All curves should decrease as SNR increases
- DNN should outperform LS at high SNR
- MMSE should be between LS and DNN

### BER vs SNR Plot
- **X-axis**: SNR in dB (0 to 40)
- **Y-axis**: Bit Error Rate (log scale)
- **Lines**: Same color scheme as MSE plot

**Expected behavior**:
- BER decreases with increasing SNR
- Better channel estimation → lower BER
- DNN should achieve lowest BER at high SNR

## Troubleshooting

### Issue 1: "Simulation file not found"
**Problem**: Missing testing simulation files

**Solution**:
1. Set `configuration = 'testing'` in [main.m](main.m#L66)
2. Set `EbN0dB = 0:5:40` in [main.m](main.m#L68)
3. Run `main` to generate all testing simulations

### Issue 2: "DNN results file not found"
**Problem**: Haven't run DNN testing

**Solution**: Run `python DNN.py High VTV_SDWW QPSK 40`

### Issue 3: "MMSE_VP_Estimate_Structure not found"
**Problem**: Old simulation files don't have MMSE estimates

**Solution**: Re-run [main.m](main.m) with the updated version to regenerate simulation files

### Issue 4: "TX_Bits_Stream_Structure not found"
**Problem**: Old simulation files don't have transmitted bits

**Solution**: Re-run [main.m](main.m) with the updated version

### Issue 5: BER is all zeros
**Problem**: Either no errors (unlikely at low SNR) or decoding issue

**Solutions**:
- Check that `Random_permutation_Vector` is saved and loaded correctly
- Verify modulation parameters match between transmission and reception
- Check that `TX_Bits_Stream_Structure` contains the original (unencoded) bits

## Advanced: Customizing the Plots

You can modify [DNN_Results_Processing_Fixed.m](DNN_Results_Processing_Fixed.m) to:

### Change Line Styles
```matlab
% Around line 218
semilogy(EbN0dB, ERR_LS, 'b--o', 'LineWidth', 3, 'MarkerSize', 10);  % Dashed line, thicker
```

### Add More Metrics
```matlab
% Calculate NMSE in dB
NMSE_LS_dB = 10*log10(ERR_LS);
NMSE_DNN_dB = 10*log10(ERR_DNN);
```

### Save Data to CSV
```matlab
% Add at the end
results_table = table(EbN0dB, ERR_LS, ERR_DNN, ERR_MMSE_VP, BER_LS, BER_DNN, BER_MMSE_VP);
writetable(results_table, 'performance_comparison.csv');
```

## Quick Command Reference

**MATLAB commands** (in order):
```matlab
% 1. Generate testing simulations (set configuration='testing' in main.m first)
main

% 2. Generate DNN testing datasets (set configuration='testing' in DNN_Datasets_Generation.m)
DNN_Datasets_Generation

% 3. (Go to terminal for Python)
% 4. Process and visualize
DNN_Results_Processing_Fixed
```

**Python command**:
```bash
# 3. Run DNN testing
python DNN.py High VTV_SDWW QPSK 40
```

## Expected Runtime

- **main.m** (testing): ~2-5 minutes per SNR point × 9 SNR points = **20-45 minutes**
- **DNN_Datasets_Generation.m**: ~1-2 minutes total
- **DNN.py** (testing): ~5-10 seconds per SNR point = **1 minute total**
- **DNN_Results_Processing_Fixed.m**: ~1-2 minutes total

**Total**: Approximately **25-50 minutes** for complete testing and visualization

## Files Generated

After completing all steps, you'll have:
- 9 simulation files (testing)
- 9 DNN dataset files (testing)
- 9 DNN results files
- 2 plot images (MSE, BER)
- 2 MATLAB figures (.fig)
- 1 results data file (.mat)

**Total storage**: ~100-500 MB depending on dataset size
