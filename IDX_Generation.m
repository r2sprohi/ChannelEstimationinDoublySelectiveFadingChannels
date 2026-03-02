clc;clearvars; close all;
IDX = 100; % total number of samples (8000 training + 2000 testing)
TR  = 0.80; % training dataset percentage (80% = 8000 samples)
TE  = 0.20; % testing dataset percentage (20% = 2000 samples)
All_IDX = 1:IDX;
training_samples = randperm(IDX,TR*IDX).'; % choose randomly 80% of the total indices for training
testing_samples = setdiff(All_IDX,training_samples).';
save(['./samples_indices_',num2str(IDX),'.mat'],  'training_samples','testing_samples'); 