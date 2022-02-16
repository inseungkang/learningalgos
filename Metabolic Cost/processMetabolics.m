clear all; clc; close all;
data = load('metabolicData.mat');
data = data.metaData;
%%
subjectPool = [1, 2, 3, 4, 5];
SessionNum = [2, 3, 4, 5];

windowSize = 180;
averageData = {};

for subject = subjectPool
    for session = SessionNum
        averageData{subject}{session}{1} = data{subject}{session}{1};
        averageData{subject}{session}{2} = movmean(data{subject}{session}{2}, windowSize);
        xvec = 1:1:length(averageData{subject}{session}{2}); xvec = xvec';
        f = fit(xvec, averageData{subject}{session}{2},'exp1');
        averageData{subject}{session}{3} = f;
        
    end
end

%% plotting
figure(1)
sgtitle('Metabolic across sessions')

subplot(5,4,1)
hold on
plot(averageData{1}{2}{2})
plot(averageData{1}{2}{3})
title('S1')
legend(strcat(num2str(averageData{1}{2}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,2)
hold on
plot(averageData{1}{3}{2})
plot(averageData{1}{3}{3})
title('S2')
legend(strcat(num2str(averageData{1}{3}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,3)
hold on
plot(averageData{1}{4}{2})
plot(averageData{1}{4}{3})
title('S3')
legend(strcat(num2str(averageData{1}{4}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,4)
hold on
plot(averageData{1}{5}{2})
plot(averageData{1}{5}{3})
title('S4')
legend(strcat(num2str(averageData{1}{5}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])


subplot(5,4,5)
hold on
plot(averageData{2}{2}{2})
plot(averageData{2}{2}{3})
legend(strcat(num2str(averageData{2}{2}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,6)
hold on
plot(averageData{2}{3}{2})
plot(averageData{2}{3}{3})
legend(strcat(num2str(averageData{2}{3}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,7)
hold on
plot(averageData{2}{4}{2})
plot(averageData{2}{4}{3})
legend(strcat(num2str(averageData{2}{4}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,8)
hold on
plot(averageData{2}{5}{2})
plot(averageData{2}{5}{3})
legend(strcat(num2str(averageData{2}{5}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])


subplot(5,4,9)
hold on
plot(averageData{3}{2}{2})
plot(averageData{3}{2}{3})
legend(strcat(num2str(averageData{3}{2}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,10)
hold on
plot(averageData{3}{3}{2})
plot(averageData{3}{3}{3})
legend(strcat(num2str(averageData{3}{3}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,11)
hold on
plot(averageData{3}{4}{2})
plot(averageData{3}{4}{3})
legend(strcat(num2str(averageData{3}{4}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,12)
hold on
plot(averageData{3}{5}{2})
plot(averageData{3}{5}{3})
legend(strcat(num2str(averageData{3}{5}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])


subplot(5,4,13)
hold on
plot(averageData{4}{2}{2})
plot(averageData{4}{2}{3})
legend(strcat(num2str(averageData{4}{2}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,14)
hold on
plot(averageData{4}{3}{2})
plot(averageData{4}{3}{3})
legend(strcat(num2str(averageData{4}{3}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,15)
hold on
plot(averageData{4}{4}{2})
plot(averageData{4}{4}{3})
legend(strcat(num2str(averageData{4}{4}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,16)
hold on
plot(averageData{4}{5}{2})
plot(averageData{4}{5}{3})
legend(strcat(num2str(averageData{4}{5}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])


subplot(5,4,17)
hold on
plot(averageData{5}{2}{2})
plot(averageData{5}{2}{3})
legend(strcat(num2str(averageData{5}{2}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,18)
hold on
plot(averageData{5}{3}{2})
plot(averageData{5}{3}{3})
legend(strcat(num2str(averageData{5}{3}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,19)
hold on
plot(averageData{5}{4}{2})
plot(averageData{5}{4}{3})
legend(strcat(num2str(averageData{5}{4}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,20)
hold on
plot(averageData{5}{5}{2})
plot(averageData{5}{5}{3})
legend(strcat(num2str(averageData{5}{5}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])