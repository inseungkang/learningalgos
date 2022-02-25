clear all; clc;
file_dir = uigetdir('','select folder containing vicon mat data');
file_dir = strcat(file_dir,'/');
% load mat file
subjectPool = [1];
sessionNum = [1, 2, 3, 4, 5];

viconData = struct;
for subject = subjectPool
    for session = sessionNum
        if session == 1
            file_list = dir(strcat(file_dir,'AB',num2str(subject),'_Session',num2str(session),'*.mat'));
            for ii = 1:length(file_list)
                filename = file_list(ii).name;
                if contains(filename,'Static') == 0
                    load(strcat(file_dir,filename));
                    
                    pelvis_marker = [markerStruct.RASI, markerStruct.LASI, markerStruct.RPSI, markerStruct.LPSI];
                    ankle_marker_r = [markerStruct.RANL, markerStruct.RANM];
                    ankle_marker_l = [markerStruct.LANL, markerStruct.LANM];
                    treadmill_r = [forceStruct.f1, forceStruct.m1, forceStruct.p1];
                    treadmill_l = [forceStruct.f2, forceStruct.m2, forceStruct.p2];
                    force_plate = [forceStruct.f3];
    
                    [com, foot_r, foot_l, force_r, force_l, fp] = resampleExtractSyncData(pelvis_marker, ankle_marker_r, ankle_marker_l, treadmill_r, treadmill_l, force_plate);
                 
                    % check treadmill speed
                    rSpeed = extractBetween(filename,'Right','_');
                    rSpeed = str2num(rSpeed{1});
                    
                    % Sync Data for 6 min walking
                    start_vec = find(fp(:,3) > 200);
                    start_idx = start_vec(1);
                    end_idx = start_idx+35999;
                    if subject == 1 && rSpeed == 12
                        end_idx = start_idx+18999;
                    end
                    
                    % save to a viconData struct
                    viconData.(strcat('AB',num2str(subject))).(strcat('Session',num2str(session))).(strcat('Speed',num2str(rSpeed))).COM = com(start_idx:end_idx,:);
                    viconData.(strcat('AB',num2str(subject))).(strcat('Session',num2str(session))).(strcat('Speed',num2str(rSpeed))).Foot_R = foot_r(start_idx:end_idx,:);
                    viconData.(strcat('AB',num2str(subject))).(strcat('Session',num2str(session))).(strcat('Speed',num2str(rSpeed))).Foot_L = foot_l(start_idx:end_idx,:);
                    viconData.(strcat('AB',num2str(subject))).(strcat('Session',num2str(session))).(strcat('Speed',num2str(rSpeed))).Force_R = force_r(start_idx:end_idx,:);
                    viconData.(strcat('AB',num2str(subject))).(strcat('Session',num2str(session))).(strcat('Speed',num2str(rSpeed))).Force_L = force_l(start_idx:end_idx,:);
                end
            end
        else
            file_list = dir(strcat(file_dir,'AB',num2str(subject),'_Session',num2str(session),'*.mat'));
            filename = file_list(1).name;
            if contains(filename,'Static') == 0
                load(strcat(file_dir,filename));
                
                pelvis_marker = [markerStruct.RASI, markerStruct.LASI, markerStruct.RPSI, markerStruct.LPSI];
                ankle_marker_r = [markerStruct.RANL, markerStruct.RANM];
                ankle_marker_l = [markerStruct.LANL, markerStruct.LANM];
                treadmill_r = [forceStruct.f1, forceStruct.m1, forceStruct.p1];
                treadmill_l = [forceStruct.f2, forceStruct.m2, forceStruct.p2];
                force_plate = [forceStruct.f3];
                
                [com, foot_r, foot_l, force_r, force_l, fp] = resampleExtractSyncData(pelvis_marker, ankle_marker_r, ankle_marker_l, treadmill_r, treadmill_l, force_plate);
                
                % check treadmill speed
                rSpeed = extractBetween(filename,'Right','_'); rSpeed = str2num(rSpeed{1});
                lSpeed = extractBetween(filename,'Left','.'); lSpeed = str2num(lSpeed{1});
                
                % Sync Data for 6 min walking
                start_vec = find(fp(:,3) > 200);
                start_idx = start_vec(1);
                start_idx = start_idx + 72000;
                end_idx = start_idx+269999;
                
                % save to a viconData struct
                viconData.(strcat('AB',num2str(subject))).(strcat('Session',num2str(session))).(strcat('rSpeed',num2str(rSpeed),'_lSpeed',num2str(lSpeed))).COM = com(start_idx:end_idx,:);
                viconData.(strcat('AB',num2str(subject))).(strcat('Session',num2str(session))).(strcat('rSpeed',num2str(rSpeed),'_lSpeed',num2str(lSpeed))).Foot_R = foot_r(start_idx:end_idx,:);
                viconData.(strcat('AB',num2str(subject))).(strcat('Session',num2str(session))).(strcat('rSpeed',num2str(rSpeed),'_lSpeed',num2str(lSpeed))).Foot_L = foot_l(start_idx:end_idx,:);
                viconData.(strcat('AB',num2str(subject))).(strcat('Session',num2str(session))).(strcat('rSpeed',num2str(rSpeed),'_lSpeed',num2str(lSpeed))).Force_R = force_r(start_idx:end_idx,:);
                viconData.(strcat('AB',num2str(subject))).(strcat('Session',num2str(session))).(strcat('rSpeed',num2str(rSpeed),'_lSpeed',num2str(lSpeed))).Force_L = force_l(start_idx:end_idx,:);
            end
        end
    end
end

save('viconData.mat','viconData');



% rSpeed = extractBetween(file_name,'Right','_');
% rSpeed = str2num(rSpeed{1});
% lSpeed = extractBetween(file_name,'Left','.');
% lSpeed = str2num(lSpeed{1});
% if rSpeed == 10
%     delta = lSpeed - rSpeed;
% else
%     delta = rSpeed - lSpeed;
% end