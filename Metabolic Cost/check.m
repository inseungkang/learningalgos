clear all; clc; close all;
data = load('metabolicData.mat');
data = data.metaData;
%%
data{2}{2}{2}(1:360)
%%
subjectPool = [1, 2, 3, 4, 5];
SessionNum = [2, 3, 4, 5];

windowSize = 180;
averageData = {};

for subject = subjectPool
    for session = SessionNum
        averageData{subject}{session}{1} = data{subject}{session}{1};
        averageData{subject}{session}{2} = mean(data{subject}{session}{2}(1:360));
        averageData{subject}{session}{3} = mean(data{subject}{session}{2}(end-360:end));

    end
end

%% plotting
figure(1)
sgtitle('Metabolic Cost change from first to last (absolute)')
x = [1, 2];
session = [2, 3, 4, 5]; 
subject = [1, 2, 3, 4, 5];

for ii = 1:20
    subplot(5,4,ii)
    a = rem(ii,5);
    if a == 0
        a = 5;
    end
    b = rem(ii,4);
    if b == 0
        b = 4;
    end
    y = [averageData{subject(a)}{session(b)}{2}-averageData{subject(a)}{session(b)}{3}];
    bar(y)
end
%%
subplot(5,4,2)
hold on
bar(averageData{1}{3}{2})
bar(averageData{1}{3}{3})
title('S2')
legend(strcat(num2str(averageData{1}{3}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,3)
hold on
bar(averageData{1}{4}{2})
bar(averageData{1}{4}{3})
title('S3')
legend(strcat(num2str(averageData{1}{4}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,4)
hold on
bar(averageData{1}{5}{2})
bar(averageData{1}{5}{3})
title('S4')
legend(strcat(num2str(averageData{1}{5}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])


subplot(5,4,5)
hold on
bar(averageData{2}{2}{2})
bar(averageData{2}{2}{3})
legend(strcat(num2str(averageData{2}{2}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,6)
hold on
bar(averageData{2}{3}{2})
bar(averageData{2}{3}{3})
legend(strcat(num2str(averageData{2}{3}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,7)
hold on
bar(averageData{2}{4}{2})
bar(averageData{2}{4}{3})
legend(strcat(num2str(averageData{2}{4}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,8)
hold on
bar(averageData{2}{5}{2})
bar(averageData{2}{5}{3})
legend(strcat(num2str(averageData{2}{5}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])


subplot(5,4,9)
hold on
bar(averageData{3}{2}{2})
bar(averageData{3}{2}{3})
legend(strcat(num2str(averageData{3}{2}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,10)
hold on
bar(averageData{3}{3}{2})
bar(averageData{3}{3}{3})
legend(strcat(num2str(averageData{3}{3}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,11)
hold on
bar(averageData{3}{4}{2})
bar(averageData{3}{4}{3})
legend(strcat(num2str(averageData{3}{4}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,12)
hold on
bar(averageData{3}{5}{2})
bar(averageData{3}{5}{3})
legend(strcat(num2str(averageData{3}{5}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])


subplot(5,4,13)
hold on
bar(averageData{4}{2}{2})
bar(averageData{4}{2}{3})
legend(strcat(num2str(averageData{4}{2}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,14)
hold on
bar(averageData{4}{3}{2})
bar(averageData{4}{3}{3})
legend(strcat(num2str(averageData{4}{3}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,15)
hold on
bar(averageData{4}{4}{2})
bar(averageData{4}{4}{3})
legend(strcat(num2str(averageData{4}{4}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,16)
hold on
bar(averageData{4}{5}{2})
bar(averageData{4}{5}{3})
legend(strcat(num2str(averageData{4}{5}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])


subplot(5,4,17)
hold on
bar(averageData{5}{2}{2})
bar(averageData{5}{2}{3})
legend(strcat(num2str(averageData{5}{2}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,18)
hold on
bar(averageData{5}{3}{2})
bar(averageData{5}{3}{3})
legend(strcat(num2str(averageData{5}{3}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,19)
hold on
bar(averageData{5}{4}{2})
bar(averageData{5}{4}{3})
legend(strcat(num2str(averageData{5}{4}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])

subplot(5,4,20)
hold on
bar(averageData{5}{5}{2})
bar(averageData{5}{5}{3})
legend(strcat(num2str(averageData{5}{5}{1}/2),'\Delta')); legend box off
set(gca,'xticklabel',[])