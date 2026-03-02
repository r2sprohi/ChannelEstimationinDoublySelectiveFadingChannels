clc; clearvars; close all; warning('off','all');

%% Configuration - MUST MATCH your training/testing setup
mobility = 'High';
ChType = 'VTV_SDWW';
modu = 'QPSK';
nUSC = 52;  % Total active subcarriers (must match main.m)
nSym = 50;  % OFDM symbols per realization
nDSC = 48;  % Data subcarriers

% Modulation parameters
if isequal(modu,'QPSK')
    nBitPerSym = 2;
elseif isequal(modu,'16QAM')
    nBitPerSym = 4;
elseif isequal(modu,'64QAM')
    nBitPerSym = 6;
end
M = 2^nBitPerSym;
Pow = mean(abs(qammod(0:(M-1), M)).^2);

% SNR range for testing
EbN0dB = (0:5:40)';
N_SNR = length(EbN0dB);

% Data subcarrier positions (excluding pilots)
% Pilots are at positions [7, 21, 32, 46] in nUSC indexing
% Data positions are all others
ppositions = [7, 21, 32, 46].';
dpositions = [1:6, 8:20, 22:31, 33:45, 47:52].';

% Coding parameters (for BER calculation)
scramInit = 93;
trellis = poly2trellis(7, [171 133]);
tbl = 34;
Interleaver_Rows = 16;
Interleaver_Columns = (nBitPerSym * nDSC * nSym) / Interleaver_Rows;

% Load the sample indices
% NOTE: Random_permutation_Vector will be loaded from each simulation file
% to ensure we use the EXACT same permutation used during transmission
load('./samples_indices_100.mat');

%% Initialize result arrays
Phf_H_Total = zeros(N_SNR, 1);

% MSE results
Err_LS = zeros(N_SNR, 1);
Err_MMSE_VP = zeros(N_SNR, 1);
Err_DNN = zeros(N_SNR, 1);

% BER results
Ber_LS = zeros(N_SNR, 1);
Ber_MMSE_VP = zeros(N_SNR, 1);
Ber_DNN = zeros(N_SNR, 1);

fprintf('Processing DNN, MMSE, and LS results...\n');
fprintf('%s\n', repmat('=', 1, 60));

%% Loop over SNR values
for n_snr = 1:N_SNR

    fprintf('Processing SNR = %d dB\n', EbN0dB(n_snr));

    %% Load simulation results (contains true channels, LS, MMSE, received symbols)
    sim_file = sprintf('./%s_%s_%s_testing_simulation_%d.mat', ...
                       mobility, ChType, modu, EbN0dB(n_snr));

    if ~exist(sim_file, 'file')
        warning('Simulation file not found: %s. Skipping SNR = %d dB', sim_file, EbN0dB(n_snr));
        continue;
    end

    load(sim_file, 'Received_Symbols_FFT_Structure', ...
                   'True_Channels_Structure', ...
                   'LS_Estimate_Structure', ...
                   'Prev_Channel_Estimate_Structure', ...
                   'MMSE_VP_Estimate_Structure', ...
                   'Random_permutation_Vector');

    % Check if MMSE results were loaded successfully
    if ~exist('MMSE_VP_Estimate_Structure', 'var')
        warning('MMSE_VP_Estimate_Structure not found in simulation file');
        MMSE_VP_Estimate_Structure = [];
    end

    % Load transmitted bits if available for BER calculation
    if exist(sim_file, 'file')
        temp = load(sim_file);
        if isfield(temp, 'TX_Bits_Stream_Structure')
            TX_Bits_Stream_Structure = temp.TX_Bits_Stream_Structure;
        else
            warning('TX_Bits_Stream_Structure not found. BER calculation will be skipped.');
            TX_Bits_Stream_Structure = [];
        end
    end

    %% Load DNN results
    dnn_file = sprintf('./%s_%s_%s_DNN_results_%d.mat', ...
                       mobility, ChType, modu, EbN0dB(n_snr));

    if ~exist(dnn_file, 'file')
        warning('DNN results file not found: %s. Skipping SNR = %d dB', dnn_file, EbN0dB(n_snr));
        continue;
    end

    load(dnn_file, 'DNN_input_X', 'DNN_true_Y', 'DNN_predicted_Y');

    %% Reshape DNN results from [samples, features] back to [nUSC, nSym, N_samples]
    % DNN_true_Y and DNN_predicted_Y are [N_samples, 2*nUSC]
    % where first nUSC columns are real, last nUSC are imaginary

    N_samples = size(DNN_true_Y, 1);
    N_ch = N_samples / nSym;  % Number of channel realizations

    % Reshape: [N_samples, 2*nUSC] -> [nUSC, nSym, N_ch]
    DNN_true_Y_complex = DNN_true_Y(:, 1:nUSC) + 1i * DNN_true_Y(:, nUSC+1:2*nUSC);
    DNN_true_Y_complex = DNN_true_Y_complex.';  % [nUSC, N_samples]
    DNN_true_Y_complex = reshape(DNN_true_Y_complex, nUSC, nSym, N_ch);

    DNN_pred_Y_complex = DNN_predicted_Y(:, 1:nUSC) + 1i * DNN_predicted_Y(:, nUSC+1:2*nUSC);
    DNN_pred_Y_complex = DNN_pred_Y_complex.';  % [nUSC, N_samples]
    DNN_pred_Y_complex = reshape(DNN_pred_Y_complex, nUSC, nSym, N_ch);

    %% Calculate MSE for each channel realization
    for u = 1:N_ch

        % True channel
        H_true = True_Channels_Structure(:, :, u);

        % LS estimate
        H_LS = LS_Estimate_Structure(:, :, u);

        % DNN estimate
        H_DNN = DNN_pred_Y_complex(:, :, u);

        % MMSE estimate (if available)
        if ~isempty(MMSE_VP_Estimate_Structure)
            H_MMSE = MMSE_VP_Estimate_Structure(:, :, u);
        end

        % Calculate channel power for normalization
        Phf_H_Total(n_snr) = Phf_H_Total(n_snr) + mean(sum(abs(H_true).^2));

        % Calculate MSE for LS
        Err_LS(n_snr) = Err_LS(n_snr) + mean(sum(abs(H_LS - H_true).^2));

        % Calculate MSE for DNN
        Err_DNN(n_snr) = Err_DNN(n_snr) + mean(sum(abs(H_DNN - H_true).^2));

        % Calculate MSE for MMSE (if available)
        if ~isempty(MMSE_VP_Estimate_Structure)
            Err_MMSE_VP(n_snr) = Err_MMSE_VP(n_snr) + mean(sum(abs(H_MMSE - H_true).^2));
        end

        %% BER Calculation (if transmitted bits are available)
        if ~isempty(TX_Bits_Stream_Structure)

            % Equalize and demodulate using LS (following main_old.m line 233)
            % Use data positions from the 52-active subcarrier set
            Bits_LS = de2bi(qamdemod(sqrt(Pow) * (Received_Symbols_FFT_Structure(dpositions, :, u) ./ H_LS(dpositions, :)), M));

            % Equalize and demodulate using DNN
            Bits_DNN = de2bi(qamdemod(sqrt(Pow) * (Received_Symbols_FFT_Structure(dpositions, :, u) ./ H_DNN(dpositions, :)), M));

            % Decode and calculate BER for LS (following main_old.m line 243)
            Ber_LS(n_snr) = Ber_LS(n_snr) + biterr(wlanScramble((vitdec(matintrlv((deintrlv(Bits_LS(:), Random_permutation_Vector)).', Interleaver_Columns, Interleaver_Rows).', trellis, tbl, 'trunc', 'hard')), scramInit), TX_Bits_Stream_Structure(:, u));

            % Decode and calculate BER for DNN
            Ber_DNN(n_snr) = Ber_DNN(n_snr) + biterr(wlanScramble((vitdec(matintrlv((deintrlv(Bits_DNN(:), Random_permutation_Vector)).', Interleaver_Columns, Interleaver_Rows).', trellis, tbl, 'trunc', 'hard')), scramInit), TX_Bits_Stream_Structure(:, u));

            % Calculate BER for MMSE (if available)
            if ~isempty(MMSE_VP_Estimate_Structure)
                Bits_MMSE = de2bi(qamdemod(sqrt(Pow) * (Received_Symbols_FFT_Structure(dpositions, :, u) ./ H_MMSE(dpositions, :)), M));
                Ber_MMSE_VP(n_snr) = Ber_MMSE_VP(n_snr) + biterr(wlanScramble((vitdec(matintrlv((deintrlv(Bits_MMSE(:), Random_permutation_Vector)).', Interleaver_Columns, Interleaver_Rows).', trellis, tbl, 'trunc', 'hard')), scramInit), TX_Bits_Stream_Structure(:, u));
            end
        end
    end

    fprintf('  Processed %d channel realizations\n', N_ch);
end

fprintf('%s\n', repmat('=', 1, 60));

%% Normalize results
N_total_samples = N_ch;  % Total number of channel realizations

% Normalize channel power
Phf_H_Total = Phf_H_Total / N_total_samples;

% Normalize MSE
ERR_LS = Err_LS ./ (N_total_samples .* Phf_H_Total);
ERR_DNN = Err_DNN ./ (N_total_samples .* Phf_H_Total);
ERR_MMSE_VP = Err_MMSE_VP ./ (N_total_samples .* Phf_H_Total);  % Always calculate (will be zeros if no MMSE data)

% Normalize BER
BER_LS = Ber_LS / (N_total_samples * nSym * nDSC * nBitPerSym);
BER_DNN = Ber_DNN / (N_total_samples * nSym * nDSC * nBitPerSym);
BER_MMSE_VP = Ber_MMSE_VP / (N_total_samples * nSym * nDSC * nBitPerSym);  % Always calculate (will be zeros if no data)

%% Plot Results

% MSE vs SNR
figure('Position', [100, 100, 800, 600]);
semilogy(EbN0dB, ERR_LS, 'b-o', 'LineWidth', 2, 'MarkerSize', 8); hold on;
semilogy(EbN0dB, ERR_DNN, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
if sum(ERR_MMSE_VP) > 0  % Check if MMSE data exists (not all zeros)
    semilogy(EbN0dB, ERR_MMSE_VP, 'g-^', 'LineWidth', 2, 'MarkerSize', 8);
    legend('LS', 'DNN', 'MMSE-VP', 'Location', 'best');
else
    legend('LS', 'DNN', 'Location', 'best');
end
grid on;
xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Normalized MSE', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('MSE vs SNR - %s, %s, %s', mobility, ChType, modu), 'FontSize', 14);
set(gca, 'FontSize', 11);

% Save MSE figure
saveas(gcf, sprintf('./%s_%s_%s_MSE_comparison.fig', mobility, ChType, modu));
saveas(gcf, sprintf('./%s_%s_%s_MSE_comparison.png', mobility, ChType, modu));

% BER vs SNR (if available)
if sum(BER_LS) > 0  % Check if BER data exists
    figure('Position', [120, 120, 800, 600]);
    semilogy(EbN0dB, BER_LS, 'b-o', 'LineWidth', 2, 'MarkerSize', 8); hold on;
    semilogy(EbN0dB, BER_DNN, 'r-s', 'LineWidth', 2, 'MarkerSize', 8);
    if sum(BER_MMSE_VP) > 0  % Check if MMSE BER data exists
        semilogy(EbN0dB, BER_MMSE_VP, 'g-^', 'LineWidth', 2, 'MarkerSize', 8);
        legend('LS', 'DNN', 'MMSE-VP', 'Location', 'best');
    else
        legend('LS', 'DNN', 'Location', 'best');
    end
    grid on;
    xlabel('SNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Bit Error Rate (BER)', 'FontSize', 12, 'FontWeight', 'bold');
    title(sprintf('BER vs SNR - %s, %s, %s', mobility, ChType, modu), 'FontSize', 14);
    set(gca, 'FontSize', 11);

    % Save BER figure
    saveas(gcf, sprintf('./%s_%s_%s_BER_comparison.fig', mobility, ChType, modu));
    saveas(gcf, sprintf('./%s_%s_%s_BER_comparison.png', mobility, ChType, modu));
end

%% Plot True Channel vs Estimates (for a sample channel realization at mid-SNR)
% Select a representative SNR (middle of range) and first channel realization
mid_snr_idx = ceil(N_SNR / 2);
sample_ch_idx = 1;

% Load the sample data
sample_sim_file = sprintf('./%s_%s_%s_testing_simulation_%d.mat', ...
                   mobility, ChType, modu, EbN0dB(mid_snr_idx));
if exist(sample_sim_file, 'file')
    load(sample_sim_file, 'True_Channels_Structure', 'LS_Estimate_Structure', 'MMSE_VP_Estimate_Structure');

    sample_dnn_file = sprintf('./%s_%s_%s_DNN_results_%d.mat', ...
                       mobility, ChType, modu, EbN0dB(mid_snr_idx));
    if exist(sample_dnn_file, 'file')
        load(sample_dnn_file, 'DNN_predicted_Y');

        % Reshape DNN prediction
        N_samples_sample = size(DNN_predicted_Y, 1);
        N_ch_sample = N_samples_sample / nSym;
        DNN_pred_complex = DNN_predicted_Y(:, 1:nUSC) + 1i * DNN_predicted_Y(:, nUSC+1:2*nUSC);
        DNN_pred_complex = DNN_pred_complex.';
        DNN_pred_complex = reshape(DNN_pred_complex, nUSC, nSym, N_ch_sample);

        % Extract sample channel realization
        H_true_sample = True_Channels_Structure(:, :, sample_ch_idx);
        H_LS_sample = LS_Estimate_Structure(:, :, sample_ch_idx);
        H_DNN_sample = DNN_pred_complex(:, :, sample_ch_idx);

        % Select a sample OFDM symbol (middle symbol)
        mid_sym_idx = ceil(nSym / 2);

        % Plot magnitude response across subcarriers
        figure('Position', [140, 140, 1000, 600]);
        subplot(2,1,1);
        plot(1:nUSC, abs(H_true_sample(:, mid_sym_idx)), 'k-', 'LineWidth', 2.5); hold on;
        plot(1:nUSC, abs(H_LS_sample(:, mid_sym_idx)), 'b--o', 'LineWidth', 1.5, 'MarkerSize', 4);
        plot(1:nUSC, abs(H_DNN_sample(:, mid_sym_idx)), 'r--s', 'LineWidth', 1.5, 'MarkerSize', 4);
        if exist('MMSE_VP_Estimate_Structure', 'var') && ~isempty(MMSE_VP_Estimate_Structure)
            H_MMSE_sample = MMSE_VP_Estimate_Structure(:, :, sample_ch_idx);
            plot(1:nUSC, abs(H_MMSE_sample(:, mid_sym_idx)), 'g--^', 'LineWidth', 1.5, 'MarkerSize', 4);
            legend('True Channel', 'LS', 'DNN', 'MMSE-VP', 'Location', 'best');
        else
            legend('True Channel', 'LS', 'DNN', 'Location', 'best');
        end
        grid on;
        xlabel('Subcarrier Index', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Channel Magnitude', 'FontSize', 12, 'FontWeight', 'bold');
        title(sprintf('Channel Frequency Response (Magnitude) - SNR = %d dB, Symbol %d', ...
              EbN0dB(mid_snr_idx), mid_sym_idx), 'FontSize', 13);
        set(gca, 'FontSize', 11);

        % Plot phase response across subcarriers
        subplot(2,1,2);
        plot(1:nUSC, angle(H_true_sample(:, mid_sym_idx)), 'k-', 'LineWidth', 2.5); hold on;
        plot(1:nUSC, angle(H_LS_sample(:, mid_sym_idx)), 'b--o', 'LineWidth', 1.5, 'MarkerSize', 4);
        plot(1:nUSC, angle(H_DNN_sample(:, mid_sym_idx)), 'r--s', 'LineWidth', 1.5, 'MarkerSize', 4);
        if exist('MMSE_VP_Estimate_Structure', 'var') && ~isempty(MMSE_VP_Estimate_Structure)
            plot(1:nUSC, angle(H_MMSE_sample(:, mid_sym_idx)), 'g--^', 'LineWidth', 1.5, 'MarkerSize', 4);
            legend('True Channel', 'LS', 'DNN', 'MMSE-VP', 'Location', 'best');
        else
            legend('True Channel', 'LS', 'DNN', 'Location', 'best');
        end
        grid on;
        xlabel('Subcarrier Index', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('Channel Phase (radians)', 'FontSize', 12, 'FontWeight', 'bold');
        title(sprintf('Channel Frequency Response (Phase) - SNR = %d dB, Symbol %d', ...
              EbN0dB(mid_snr_idx), mid_sym_idx), 'FontSize', 13);
        set(gca, 'FontSize', 11);

        % Save channel comparison figure
        saveas(gcf, sprintf('./%s_%s_%s_channel_comparison.fig', mobility, ChType, modu));
        saveas(gcf, sprintf('./%s_%s_%s_channel_comparison.png', mobility, ChType, modu));
    end
end

%% Save numerical results
save(sprintf('./%s_%s_%s_comparison_results.mat', mobility, ChType, modu), ...
     'EbN0dB', 'ERR_LS', 'ERR_DNN', 'ERR_MMSE_VP', ...
     'BER_LS', 'BER_DNN', 'BER_MMSE_VP');

fprintf('\nResults saved successfully!\n');
fprintf('MSE plot: %s_%s_%s_MSE_comparison.png\n', mobility, ChType, modu);
if sum(BER_LS) > 0
    fprintf('BER plot: %s_%s_%s_BER_comparison.png\n', mobility, ChType, modu);
end
fprintf('Channel comparison plot: %s_%s_%s_channel_comparison.png\n', mobility, ChType, modu);
fprintf('Data: %s_%s_%s_comparison_results.mat\n', mobility, ChType, modu);
