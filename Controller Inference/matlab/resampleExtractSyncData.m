function [com, foot_r, foot_l, force_r, force_l, fp] = resampleExtractSyncData(pelvis_marker, ankle_marker_r, ankle_marker_l, treadmill_r, treadmill_l, force_plate)
    COM_pos_X = mean([pelvis_marker(:,1), pelvis_marker(:,4), pelvis_marker(:,7), pelvis_marker(:,10)], 2);
    COM_pos_Y = mean([pelvis_marker(:,2), pelvis_marker(:,5), pelvis_marker(:,8), pelvis_marker(:,11)], 2);
    COM_pos_Z = mean([pelvis_marker(:,3), pelvis_marker(:,6), pelvis_marker(:,9), pelvis_marker(:,12)], 2);
    
    Foot_r_X = mean([ankle_marker_r(:,1), ankle_marker_r(:,4)], 2);
    Foot_r_Y = mean([ankle_marker_r(:,2), ankle_marker_r(:,5)], 2);
    Foot_r_Z = mean([ankle_marker_r(:,3), ankle_marker_r(:,6)], 2);
    
    Foot_l_X = mean([ankle_marker_l(:,1), ankle_marker_l(:,4)], 2);
    Foot_l_Y = mean([ankle_marker_l(:,2), ankle_marker_l(:,5)], 2);
    Foot_l_Z = mean([ankle_marker_l(:,3), ankle_marker_l(:,6)], 2);

    com = [COM_pos_X, COM_pos_Y, COM_pos_Z];
    foot_r = [Foot_r_X, Foot_r_Y, Foot_r_Z];
    foot_l = [Foot_l_X, Foot_l_Y, Foot_l_Z];

    % Resampling force data into 100 Hz
    force_r = []; force_l = []; fp = [];
    for ii = 1:length(treadmill_r)/10
        force_r(ii,1:9) = treadmill_r(ii*10-9,1:9);
        force_l(ii,1:9) = treadmill_l(ii*10-9,1:9);
        fp(ii,1:3) = force_plate(ii*10-9,1:3);
    end