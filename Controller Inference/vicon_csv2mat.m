function vicon_csv2mat(filename)
    % load data from VICON csv file
    opts = delimitedTextImportOptions;
    data = readtable(filename,opts);
    
    % extract subject number from the filename
    subjectNum = extractBetween(filename,'AB','_');
    
    % obtain index number for markers
    marker_index = find(strcmp(data.Var1, 'Trajectories'))+5;
    RPSI_col_index = find(strcmp(data{marker_index-3,:}, strcat('AB',subjectNum,':RPSI')));
    LPSI_col_index = find(strcmp(data{marker_index-3,:}, strcat('AB',subjectNum,':LPSI')));
    RASI_col_index = find(strcmp(data{marker_index-3,:}, strcat('AB',subjectNum,':RASI')));
    LASI_col_index = find(strcmp(data{marker_index-3,:}, strcat('AB',subjectNum,':LASI')));
    RANL_col_index = find(strcmp(data{marker_index-3,:}, strcat('AB',subjectNum,':RANL')));
    RANM_col_index = find(strcmp(data{marker_index-3,:}, strcat('AB',subjectNum,':RANM')));
    LANL_col_index = find(strcmp(data{marker_index-3,:}, strcat('AB',subjectNum,':LANL')));
    LANM_col_index = find(strcmp(data{marker_index-3,:}, strcat('AB',subjectNum,':LANM')));
    
    % extract data from marker indicies
    pelvis_marker = [data{marker_index:end,RPSI_col_index:RPSI_col_index+2},...
                     data{marker_index:end,LPSI_col_index:LPSI_col_index+2},...
                     data{marker_index:end,RASI_col_index:RASI_col_index+2},...
                     data{marker_index:end,LASI_col_index:LASI_col_index+2}];
    
    ankle_marker_r = [data{marker_index:end,RANL_col_index:RANL_col_index+2},...
                      data{marker_index:end,RANM_col_index:RANM_col_index+2}];
    
    ankle_marker_l = [data{marker_index:end,LANL_col_index:LANL_col_index+2},...
                      data{marker_index:end,LANM_col_index:LANM_col_index+2}];
    
    % extract force plate and treadmill data
    force_index = 6;
    tread_r_col_index = find(strcmp(data{3,:}, 'RightTreadmillBelt - Force'));
    tread_l_col_index = find(strcmp(data{3,:}, 'LeftTreadmillBelt - Force'));
    force_col_idx = find(strcmp(data{3,:}, 'LeftPlate - Force'));
    
    treadmill_r = data{force_index:marker_index-6,tread_r_col_index:tread_r_col_index+8};
    treadmill_l = data{force_index:marker_index-6,tread_l_col_index:tread_l_col_index+8};
    force_plate = data{force_index:marker_index-6,force_col_idx:force_col_idx+2};
    
    % convert string data (cell) into a double array
    pelvis_marker = cellfun(@str2num, pelvis_marker);
    ankle_marker_r = cellfun(@str2num, ankle_marker_r);
    ankle_marker_l = cellfun(@str2num, ankle_marker_l);
    treadmill_r = cellfun(@str2num, treadmill_r);
    treadmill_l = cellfun(@str2num, treadmill_l);
    force_plate = cellfun(@str2num, force_plate);
    
    % save data into .mat structure
    saving_filename = extractBefore(filename,'.csv');
    save(strcat(saving_filename,'.mat'),'pelvis_marker','ankle_marker_r','ankle_marker_l','force_plate','treadmill_r','treadmill_l');

end