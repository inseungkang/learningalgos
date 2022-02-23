clear all; clc; close all;
% check directory
file_dir = uigetdir('','select folder containing vicon csv data');

% obtain file name in a list and convert csv to mat file
fnames = {dir(file_dir).name};
for ii = 1:length(fnames)
    if contains(fnames{ii}, '.csv') == 1
        viconConvCSV2MAT(strcat(file_dir,'/',fnames{ii}))
    end
end
