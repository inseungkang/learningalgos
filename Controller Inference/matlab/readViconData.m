%%
clear all; clc;
filename = 'D:\Vicon Data_CSV\Vicon Data_CSV\AB1_Session1_Right6_Left6.csv';
opts = delimitedTextImportOptions;
data = readtable(filename,opts);

%%
subjectNum = extractBetween(filename,'AB','_');

% obtain index number for markers
marker_index = find(strcmp(data.Var1, 'Trajectories'))+5;
RPSI_col_index = find(strcmp(data{marker_index-3,:}, strcat('AB',subjectNum,':RPSI')));
LPSI_col_index = find(strcmp(data{marker_index-3,:}, strcat('AB',subjectNum,':LPSI')));
RASI_col_index = find(strcmp(data{marker_index-3,:}, strcat('AB',subjectNum,':RASI')));
LASI_col_index = find(strcmp(data{marker_index-3,:}, strcat('AB',subjectNum,':LASI')));

pelvis_marker = [data{marker_index:end,RPSI_col_index:RPSI_col_index+2},...
                 data{marker_index:end,LPSI_col_index:LPSI_col_index+2},...
                 data{marker_index:end,RASI_col_index:RASI_col_index+2},...
                 data{marker_index:end,LASI_col_index:LASI_col_index+2}];
pelvis_marker(cellfun(@isempty,pelvis_marker)) = {'0'};
%%
COM_pos_X = mean([pelvis_marker(:,1), pelvis_marker(:,4), pelvis_marker(:,7), pelvis_marker(:,10)], 2);
COM_pos_Y = mean([pelvis_marker(:,2), pelvis_marker(:,5), pelvis_marker(:,8), pelvis_marker(:,11)], 2);
COM_pos_Z = mean([pelvis_marker(:,3), pelvis_marker(:,6), pelvis_marker(:,9), pelvis_marker(:,12)], 2);

Foot_r_X = mean([ankle_marker_r(:,1), ankle_marker_r(:,4)], 2);
Foot_r_Y = mean([ankle_marker_r(:,2), ankle_marker_r(:,5)], 2);
Foot_r_Z = mean([ankle_marker_r(:,3), ankle_marker_r(:,6)], 2);

Foot_l_X = mean([ankle_marker_l(:,1), ankle_marker_l(:,4)], 2);
Foot_l_Y = mean([ankle_marker_l(:,2), ankle_marker_l(:,5)], 2);
Foot_l_Z = mean([ankle_marker_l(:,3), ankle_marker_l(:,6)], 2);

% Resampling force data into 100 Hz
Force_r = []; Force_l = []; FP = [];
for ii = 1:length(treadmill_r)/10
    Force_r(ii,1:9) = treadmill_r(ii*10-9,1:9);
    Force_l(ii,1:9) = treadmill_l(ii*10-9,1:9);
    FP(ii,1:3) = force_plate(ii*10-9,1:3);
end

%% Sync Data
start_vec = find(FP(:,3) < -200);
start_idx = start_vec(1);

COM_pos_X = COM_pos_X(start_idx:start_idx+36000);
COM_pos_Y = COM_pos_Y(start_idx:start_idx+36000);
COM_pos_Z = COM_pos_Z(start_idx:start_idx+36000);

Foot_r_X = Foot_r_X(start_idx:start_idx+36000);
Foot_r_Y = Foot_r_Y(start_idx:start_idx+36000);
Foot_r_Z = Foot_r_Z(start_idx:start_idx+36000);

Foot_l_X = Foot_l_X(start_idx:start_idx+36000);
Foot_l_Y = Foot_l_Y(start_idx:start_idx+36000);
Foot_l_Z = Foot_l_Z(start_idx:start_idx+36000);

Force_r = Force_r(start_idx:start_idx+36000,:);
Force_l = Force_l(start_idx:start_idx+36000,:);
%% Computing gait phase for both legs
index_vec = Force_r(:,3) > -50;
index_vec = find(diff(index_vec) == -1);

GP_r = zeros(index_vec(1)-1,1);
for ii = 1:length(index_vec)-1
    gp_vec = linspace(0,1,index_vec(ii+1)-index_vec(ii)+1)';
    GP_r = [GP_r; gp_vec(1:end-1)];
end
GP_r = [GP_r; zeros(length(Force_r(:,3))-index_vec(end)+1,1)];

index_vec = Force_l(:,3) > -50;
index_vec = find(diff(index_vec) == -1);

GP_l = zeros(index_vec(1)-1,1);
for ii = 1:length(index_vec)-1
    gp_vec = linspace(0,1,index_vec(ii+1)-index_vec(ii)+1)';
    GP_l = [GP_l; gp_vec(1:end-1)];
end
GP_l = [GP_l; zeros(length(Force_r(:,3))-index_vec(end)+1,1)];

%% Obtaining gait event index (start, end) and plot raw GRF and foot Z
index_vec = Force_r(:,3) > -50;
start = find(diff(index_vec) == -1); stop = find(diff(index_vec) == 1);
GP_index = [start(1:end-1), stop(2:end)+1];

x1 = 1:length(Force_r(:,3));
y1 = Force_r(:,3);

close all
subplot(2,1,1)
hold on
plot(x1, y1, 'o-','Color','blue','MarkerIndices', GP_index(:,1),'MarkerEdgeColor','red','linewidth',2,'MarkerSize',10)
plot(x1, y1, 'o-','Color','blue','MarkerIndices', GP_index(:,2),'MarkerEdgeColor','green','linewidth',2,'MarkerSize',10)
title('Vertical GRF with gait events')
xlabel('Time in every 10 ms')
ylabel('Force in N')
xlim([5000 5500])

subplot(2,1,2)
x2 = 1:length(Foot_r_Z);
y2 = Foot_r_Z;

hold on
plot(x2, y2, 'o-','Color','blue','MarkerIndices', GP_index(:,1),'MarkerEdgeColor','red','linewidth',2,'MarkerSize',10)
plot(x2, y2, 'o-','Color','blue','MarkerIndices', GP_index(:,2),'MarkerEdgeColor','green','linewidth',2,'MarkerSize',10)
title('Vertical foot position with gait events')
xlabel('Time in every 10 ms)')
ylabel('position in mm')
xlim([5000 5500])

%% Plot foot placement overtime

x1 = 1:length(Force_r(:,3));
y1 = Force_r(:,3);

footPlacement = []; 

for ii = 1:length(GP_index)
    footPlacement = [footPlacement; [GP_index(ii) Foot_r_X(GP_index(ii,1))-COM_pos_X(GP_index(ii,1)) Foot_r_Y(GP_index(ii,1))-COM_pos_Y(GP_index(ii,1))]];
end

subplot(2,1,1)
plot(footPlacement(:,2), 'o')
title('X axis foot placement over steps')
ylabel('Position in mm')
subplot(2,1,2)
plot(footPlacement(:,3), 'o')
title('Y axis foot placement over steps')
ylabel('Position in mm')
xlabel('Steps taken')
%%

close all
subplot(2,1,1)
hold on
plot(x1, y1, 'o-','Color','blue','MarkerIndices', GP_index(:,1),'MarkerEdgeColor','red','linewidth',2,'MarkerSize',10)
plot(x1, y1, 'o-','Color','blue','MarkerIndices', GP_index(:,2),'MarkerEdgeColor','green','linewidth',2,'MarkerSize',10)
title('Vertical GRF with gait events')
xlabel('Time in every 10 ms)')
ylabel('Force in N')
xlim([5000 5500])

subplot(2,1,2)
x2 = 1:length(Foot_r_Z);
y2 = Foot_r_Z;

hold on
plot(x2, y2, 'o-','Color','blue','MarkerIndices', GP_index(:,1),'MarkerEdgeColor','red','linewidth',2,'MarkerSize',10)
plot(x2, y2, 'o-','Color','blue','MarkerIndices', GP_index(:,2),'MarkerEdgeColor','green','linewidth',2,'MarkerSize',10)
title('Vertical foot position with gait events')
xlabel('Time in every 10 ms)')
ylabel('position in mm')
xlim([5000 5500])

%% Sync Data
start_vec = find(force_plate(:,3) < -200);
start_idx = start_vec(1);

COM_pos_X = COM_pos_X(start_idx:end);
COM_pos_Y = COM_pos_Y(start_idx:end);
COM_pos_Z = COM_pos_Z(start_idx:end);

Foot_r_X = Foot_r_X(start_idx:end);
Foot_r_Y = Foot_r_Y(start_idx:end);
Foot_r_Z = Foot_r_Z(start_idx:end);

Foot_l_X = Foot_l_X(start_idx:end);
Foot_l_Y = Foot_l_Y(start_idx:end);
Foot_l_Z = Foot_l_Z(start_idx:end);

GP_r = GP_r(start_idx:end);
GP_l = GP_l(start_idx:end);

%% Foot Placement
foot_placement = [];
for ii = 1:length(GP_r)
    if GP_r(ii) == 0
        foot_placement = [foot_placement; [Foot_r_X(ii), Foot_r_Y(ii)]];
    end
end
step_vec = 1:length(foot_placement);
plot3(step_vec,foot_placement(:,1),foot_placement(:,2))

%%
step_width_vec = [];
step_length_vec = [];

for ii = 1:length(GP_r)
    if GP_r(ii) == 0
        step_width_vec = [step_width_vec; Foot_r_X(ii) - Foot_l_X(ii)];
        step_length_vec = [step_length_vec; Foot_r_Y(ii) - Foot_l_Y(ii)];
    end
end

subplot(2,1,1)
plot(step_width_vec)
subplot(2,1,2)
plot(step_length_vec)
%%
figure(1)
hold on
subplot(3,1,1)
plot(COM_pos_Z)

subplot(3,1,2)
plot(Foot_r_Z)

subplot(3,1,3)
plot(Foot_l_Z)


