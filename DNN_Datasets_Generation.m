clc; clearvars; close all; warning('off','all');

%% Load predefined indices
load('./samples_indices_100.mat');
configuration = 'testing';   % 'training' or 'testing'

%% System parameters
% IMPORTANT: nUSC MUST match the value from main.m (nDSC + nPSC = 48 + 4 = 52)
nUSC     = 52;       % active subcarriers (pilots + data)
nSym     = 50;       % OFDM symbols per realization
mobility = 'High';
modu     = 'QPSK';
ChType   = 'VTV_SDWW';

%% Validate that nUSC is consistent
% This should match: input_dim = 6 * nUSC, output_dim = 2 * nUSC in Python
fprintf('Dataset generation parameters:\n');
fprintf('  nUSC = %d (input_dim will be %d, output_dim will be %d)\n', nUSC, 6*nUSC, 2*nUSC);
fprintf('  Configuration: %s\n', configuration);
fprintf('  Mobility: %s, Channel: %s, Modulation: %s\n\n', mobility, ChType, modu);

%% SNR selection
if isequal(configuration,'training')
    indices = training_samples;
    EbN0dB  = 40;
else
    indices = testing_samples;
    EbN0dB  = 0:5:40;
end

Dataset_size = size(indices,1);
N_SNR        = length(EbN0dB);

%% Loop over SNR points
for n_snr = 1:N_SNR

    %% Load simulation results (UPDATED FILE CONTENTS)
    load(['./',mobility,'_',ChType,'_',modu,'_',configuration,...
          '_simulation_' num2str(EbN0dB(n_snr)),'.mat'], ...
          'Received_Symbols_FFT_Structure', ...
          'LS_Estimate_Structure', ...
          'Prev_Channel_Estimate_Structure', ...
          'True_Channels_Structure');

    %% Validate loaded data dimensions
    if size(Received_Symbols_FFT_Structure, 1) ~= nUSC
        error('Dimension mismatch! Expected nUSC=%d, got %d', nUSC, size(Received_Symbols_FFT_Structure, 1));
    end
    if size(Received_Symbols_FFT_Structure, 2) ~= nSym
        error('Dimension mismatch! Expected nSym=%d, got %d', nSym, size(Received_Symbols_FFT_Structure, 2));
    end

    %% Reshape: [nUSC × nSym × Nch] → [nUSC × (nSym·Nch)]
    Y_exp   = reshape(Received_Symbols_FFT_Structure, ...
                      nUSC, nSym * Dataset_size);

    HLS_exp = reshape(LS_Estimate_Structure, ...
                      nUSC, nSym * Dataset_size);

    HPR_exp = reshape(Prev_Channel_Estimate_Structure, ...
                      nUSC, nSym * Dataset_size);

    HT_exp  = reshape(True_Channels_Structure, ...
                      nUSC, nSym * Dataset_size);

    %% Allocate DL tensors
    NumSamples = nSym * Dataset_size;
    Dataset_X  = zeros(6*nUSC, NumSamples);
    Dataset_Y  = zeros(2*nUSC, NumSamples);

    %% Construct DL inputs (complex → real)
    Dataset_X(1:nUSC, :)           = real(Y_exp);
    Dataset_X(nUSC+1:2*nUSC, :)    = imag(Y_exp);

    Dataset_X(2*nUSC+1:3*nUSC, :)  = real(HLS_exp);
    Dataset_X(3*nUSC+1:4*nUSC, :)  = imag(HLS_exp);

    Dataset_X(4*nUSC+1:5*nUSC, :)  = real(HPR_exp);
    Dataset_X(5*nUSC+1:6*nUSC, :)  = imag(HPR_exp);

    %% Construct labels
    Dataset_Y(1:nUSC, :)        = real(HT_exp);
    Dataset_Y(nUSC+1:2*nUSC, :) = imag(HT_exp);

    %% Assign to train/test structures
    if isequal(configuration,'training')
        DNN_Datasets.Train_X = Dataset_X.';
        DNN_Datasets.Train_Y = Dataset_Y.';
    else
        DNN_Datasets.Test_X  = Dataset_X.';
        DNN_Datasets.Test_Y  = Dataset_Y.';
    end

    %% Save dataset (filename format matches Python DNN.py expectations)
    save(['./',mobility,'_',ChType,'_',modu,...
          '_DNN_',configuration,'_dataset_' num2str(EbN0dB(n_snr)),'.mat'], ...
          'DNN_Datasets');

    disp(['Saved: ',mobility,'_',ChType,'_',modu,'_DNN_',configuration,'_dataset_' num2str(EbN0dB(n_snr)),'.mat']);

end
