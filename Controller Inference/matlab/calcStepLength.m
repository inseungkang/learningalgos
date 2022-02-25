clear all; clc;
load viconData.mat;

% Session 1
subjectNum = [1]; sessionNum = [1]; speedNum = [6, 8, 10, 12, 14];

stepLength = struct;
for subject = subjectNum
    for session = sessionNum
        for speed = speedNum
            data = viconData.(strcat('AB',num2str(subject))).(strcat('Session',num2str(session))).(strcat('Speed',num2str(speed)));

        end
    end
end
%%
start = find(diff(data.Force_R(:,3) > 50) == 1); stop = find(diff(data.Force_R(:,3) > 50) == -1);
stance_idx = [start, stop+1];
sl_r = [];
for ii = 1:length(stance_idx)
    sl_r = [sl_r; data.Foot_R(stance_idx(ii),2) - data.Foot_L(stance_idx(ii),2)];
end

%%
% x1 = 1:length(data.Force_R(:,3));
% y1 = data.Force_R(:,3);

close all
subplot(2,1,1)
hold on
plot(stance_idx(:,1), sl_r, 'o-','Color','blue','MarkerIndices', stance_index(:,1),'MarkerEdgeColor','red','linewidth',2,'MarkerSize',10)
plot(x1, y1, 'o-','Color','blue','MarkerIndices', stance_index(:,2),'MarkerEdgeColor','green','linewidth',2,'MarkerSize',10)
title('Vertical GRF with gait events')
xlabel('Time in every 10 ms')
ylabel('Force in N')
xlim([1 2000])
%%
close all
plot(data.Force_R(10000:20000,3))
