% Check what's in the MAT file
matfile = 'High_VTV_SDWW_QPSK_testing_simulation_0.mat';
vars = whos('-file', matfile);
fprintf('Variables in %s:\n', matfile);
for i = 1:length(vars)
    fprintf('  %s: %s\n', vars(i).name, mat2str(vars(i).size));
end
