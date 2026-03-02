clc; clearvars; close all; warning('off','all');

ch_func = Channel_functions();

%% ================= OFDM PARAMETERS (IEEE 802.11p) =================
ofdmBW   = 10e6;
nFFT     = 64;
nDSC     = 48;       % Data subcarriers
nPSC     = 4;        % Pilot subcarriers
nZSC     = 12;       % Zero (null) subcarriers
nUSC     = nDSC + nPSC;  % Total active subcarriers = 52
                         % IMPORTANT: This must match DNN_Datasets_Generation.m
                         % DNN input_dim = 6*nUSC = 312
                         % DNN output_dim = 2*nUSC = 104
K        = nUSC + nZSC;
nSym     = 50;       % OFDM symbols per channel realization

deltaF  = ofdmBW/nFFT; 
Tfft    = 1/deltaF;
Tgi     = Tfft/4; %tgi > channel's max delay spread
K_cp    = nFFT*Tgi/Tfft; %cyclic prefix length

pilots_locations = [8 22 44 58].'; %subcarrier index at which it is present
pilots           = [1 1 1 -1].';
data_locations   = [2:7 9:21 23:27 39:43 45:57 59:64].';
ppositions       = [7 21 32 46].';%location in time domain
dpositions       = [1:6 8:20 22:31 33:45 47:52].';

%% ================= PREAMBLE =================
dp = [0 0 0 0 0 0 1 1 -1 -1 1 1 -1 1 -1 1 1 1 1 1 1 -1 -1 1 ...
      1 -1 1 -1 1 1 1 1 0 1 -1 -1 1 1 -1 1 -1 1 -1 -1 -1 -1 -1 ...
      1 1 -1 -1 1 -1 1 -1 1 1 1 1 0 0 0 0 0];
dp = fftshift(dp);
Ep = 1;
Kset = find(dp ~= 0);
Kon  = length(Kset);

dp = sqrt(Ep) * dp.';
xp = sqrt(K) * ifft(dp);
xp_cp = [xp(end-K_cp+1:end); xp];
preamble_80211p = repmat(xp_cp,1,2);

%% ================= MODULATION =================
modu = 'QPSK';  % Modulation scheme - must match DNN_Datasets_Generation.m
                % and DNN.py (modulation parameter)
nBitPerSym = 2;
M = 4;
Pow = mean(abs(qammod(0:M-1,M)).^2);

%% ================= CODING =================
scramInit = 93;
trellis   = poly2trellis(7,[171 133]);
tbl       = 34;
rate      = 1/2;

Interleaver_Rows = 16;
Interleaver_Columns = (nBitPerSym*nDSC*nSym)/Interleaver_Rows;
Random_permutation_Vector = randperm(nBitPerSym*nDSC*nSym);

%% ================= CHANNEL =================
ChType = 'VTV_SDWW';  % Channel model - must match DNN_Datasets_Generation.m
                       % and DNN.py (channel_model parameter)
fs = K*deltaF;
fD = 500;
rchan = ch_func.GenFadingChannel(ChType,fD,fs); %fading channel object

%% ================= SIMULATION SETUP =================
load('./samples_indices_100.mat');
configuration = 'testing';  % 'training' or 'testing' - must match DNN_Datasets_Generation.m
indices = testing_samples;
EbN0dB = 0:5:40;  % For training, use single SNR; for testing, use 0:5:40

SNR_p = EbN0dB + 10*log10(K/nDSC) + 10*log10(K/(K+K_cp)) ...
                  + 10*log10(nBitPerSym) + 10*log10(rate);
SNR_p = SNR_p.';
N0 = Ep*10.^(-SNR_p/10);

N_CH = size(indices,1);
N_SNR = length(EbN0dB);

Err_LS = zeros(N_SNR,1);
Err_MMSE_VP = zeros(N_SNR,1);
Ber_Ideal = zeros(N_SNR,1);
Ber_LS = zeros(N_SNR,1);
Ber_MMSE_VP = zeros(N_SNR,1);
Phf_H_Total = zeros(N_SNR,1);

%% ================= MAIN LOOP =================
for n_snr = 1:N_SNR
    disp(['Running SNR = ', num2str(EbN0dB(n_snr))]);
    tic;
    %Kon - active subcarriers , nsym (ofdm symbols per packet) and N_CH channel realizations (N_CH = number of monte carlo simulations)
    Received_Symbols_FFT_Structure = zeros(Kon,nSym,N_CH);
    True_Channels_Structure = zeros(Kon,nSym,N_CH);
    LS_Estimate_Structure = zeros(Kon,nSym,N_CH);
    Prev_Channel_Estimate_Structure = zeros(Kon,nSym,N_CH);
    MMSE_VP_Estimate_Structure = zeros(Kon,nSym,N_CH);
    TX_Bits_Stream_Structure = zeros(nDSC*nSym*nBitPerSym*rate,N_CH);

    for n_ch = 1:N_CH
        %% ----- TRANSMITTER -----
        bits = randi([0 1], nDSC*nSym*nBitPerSym*rate,1);
        bits_scr = wlanScramble(bits,scramInit);
        bits_enc = convenc(bits_scr,trellis);

        bits_int = matintrlv(bits_enc.',Interleaver_Rows,Interleaver_Columns).';
        bits_int = intrlv(bits_int,Random_permutation_Vector);

        bits_reshaped = reshape(bits_int,nDSC,nSym,nBitPerSym);
        sym_idx = bits_reshaped(:,:,1) + 2*bits_reshaped(:,:,2);
        symbols = qammod(sym_idx,M)/sqrt(Pow);

        OFDM = zeros(K,nSym);
        OFDM(data_locations,:) = symbols;
        OFDM(pilots_locations,:) = repmat(pilots,1,nSym);

        tx = sqrt(K)*ifft(OFDM);
        tx = [tx(end-K_cp+1:end,:); tx];
        tx = [preamble_80211p tx];

        %% ----- CHANNEL -----
        release(rchan);
        rchan.Seed = indices(n_ch,1);
        [h,y] = ch_func.ApplyChannel(rchan,tx,K_cp);

        yp = y(K_cp+1:end,1:2);
        y = y(K_cp+1:end,3:end);

        yfp = sqrt(1/K)*fft(yp);
        yFD = sqrt(1/K)*fft(y);

        h = h(K_cp+1:end,:);
        hf = fft(h);
        hf = hf(:,3:end);

        Phf_H_Total(n_snr) = Phf_H_Total(n_snr) + mean(sum(abs(hf(Kset,:)).^2));

        %% ----- NOISE -----
        yfp = yfp + sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,2],1);
        yFD = yFD + sqrt(N0(n_snr))*ch_func.GenRandomNoise(size(yFD),1);

        %% ----- CAUSAL LS TRACKING -----
        he_LS = (yfp(Kset,1)+yfp(Kset,2)) ./ (2*dp(Kset));
        H_LS = zeros(Kon,nSym);
        H_LS(:,1) = he_LS;

        eps_val = 1e-8;
        for n = 2:nSym
            Xhat = yFD(Kset,n)./(H_LS(:,n-1)+eps_val);
            Shat = qammod(qamdemod(sqrt(Pow)*Xhat,M),M)/sqrt(Pow);
            H_LS(:,n) = yFD(Kset,n)./(Shat+eps_val);
        end

        LS_Estimate_Structure(:,:,n_ch) = H_LS;
        Prev_Channel_Estimate_Structure(:,1,n_ch) = H_LS(:,1);
        Prev_Channel_Estimate_Structure(:,2:end,n_ch) = H_LS(:,1:end-1);

        Err_LS(n_snr) = Err_LS(n_snr) + mean(sum(abs(H_LS - hf(Kset,:)).^2));

        %% ----- MMSE-VP -----
        [H_MMSE_VP,EQ_MMSE] = MMSE_Virtual_Pilots(he_LS,yFD,Kset,modu,ppositions,dpositions,var(yFD(:)));
        Err_MMSE_VP(n_snr) = Err_MMSE_VP(n_snr) + mean(sum(abs(H_MMSE_VP - hf(Kset,:)).^2));

        %% ----- SAVE STRUCTURES -----
        Received_Symbols_FFT_Structure(:,:,n_ch) = yFD(Kset,:);
        True_Channels_Structure(:,:,n_ch) = hf(Kset,:);
        MMSE_VP_Estimate_Structure(:,:,n_ch) = H_MMSE_VP;
        TX_Bits_Stream_Structure(:,n_ch) = bits;
    end
    if isequal(configuration,'testing')
        save(['./High_VTV_SDWW_QPSK_testing_simulation_' num2str(EbN0dB(n_snr))], ...
            'Received_Symbols_FFT_Structure', ...
            'True_Channels_Structure', ...
            'LS_Estimate_Structure', ...
            'Prev_Channel_Estimate_Structure', ...
            'MMSE_VP_Estimate_Structure', ...
            'TX_Bits_Stream_Structure', ...
            'Random_permutation_Vector');
    else
        save(['./High_VTV_SDWW_QPSK_training_simulation_' num2str(EbN0dB(n_snr))], ...
            'Received_Symbols_FFT_Structure', ...
            'True_Channels_Structure', ...
            'LS_Estimate_Structure', ...
            'Prev_Channel_Estimate_Structure', ...
            'MMSE_VP_Estimate_Structure', ...
            'TX_Bits_Stream_Structure', ...
            'Random_permutation_Vector');
    end

    toc;
end

