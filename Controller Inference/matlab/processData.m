clear all; clc; close all;
load viconData.mat;

subjectNum = [1, 2, 3, 4];
sessionNum = [1, 2, 3, 4, 5];
speedNum = [6, 8, 10, 12, 14];

% figure(1)
% xlim([0 36000])
% hold on

for subject = subjectNum
    for session = sessionNum
        if session = 1
            for speed = speedNum
                data = viconData.(strcat('AB',num2str(subject))).(strcat('Session',num2str(session))).(strcat('Speed',num2str(speed)));
                rightSpeed = speed; leftSpeed = speed;
                viconData.(strcat('AB',num2str(subject))).(strcat('Session',num2str(session))).(strcat('Speed',num2str(speed))).SLA = calcStepLength(data, rightSpeed, leftSpeed);
                SLA = viconData.(strcat('AB',num2str(subject))).(strcat('Session',num2str(session))).(strcat('Speed',num2str(speed))).SLA;
            end

        else
            for

                
        end
    end
end






%%
heelstrike_r = find(diff(data.Force_R(:,3) > 200) == 1);
heelstrike_l = find(diff(data.Force_L(:,3) > 200) == 1);

if length(heelstrike_r) > length(heelstrike_l)
    heelstrike_r = heelstrike_r(1:length(heelstrike_l));
else
    heelstrike_l = heelstrike_l(1:length(heelstrike_r));
end

test = [heelstrike_r, heelstrike_l];
%%
close all
plot(data.Force_R(:,3))