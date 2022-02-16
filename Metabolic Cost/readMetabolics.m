clear all; close all; clc;

current_dir = pwd;
data_dir = strcat(current_dir, '/Data/');

startidx = 721;
stopidx = 2700+721-1;

subjectPool = [1, 2, 3, 4, 5];
SessionNum = [2, 3, 4, 5];
massPool = [0, 68.7, 68.7, 67, 67.2;...
            0, 62, 59, 59.8, 60.3;...
            0, 59.6, 58.2, 58.4, 58.3;...
            0, 77.3, 77.7, 78.4, 78.2;...
            0, 59.5, 59.8, 59.8, 60.3];

metaData = {};
     
%%
for subject = subjectPool
    for session = SessionNum
        bodymass = massPool(subject, session);

        file_name = dir(strcat(data_dir,'AB',num2str(subject),'_Metabolics_Session',num2str(session),'*.csv')).name;
        data = readtable(strcat(data_dir,file_name));
        file_name
        rSpeed = extractBetween(file_name,'Right','_');
        rSpeed = str2num(rSpeed{1});
        lSpeed = extractBetween(file_name,'Left','.');
        lSpeed = str2num(lSpeed{1});
        if rSpeed == 10
            delta = lSpeed - rSpeed;
        else
            delta = rSpeed - lSpeed;
        end

% Check string format timestamp and convert to time in seconds 
        original = data.Var1;
        for ii = 1:length(original)
            convStr = split(original{ii},':');
            
            if length(convStr) == 3 && str2num(convStr{1}) > 1
                strvec = original{ii};
                original{ii} = strvec(1:5);
            end
        end

        time = [];
        for ii = 1:length(original)
            current_string = split(original{ii},':');
        
            if length(current_string) == 3
                time(ii) = str2num(current_string{1})*3600 + str2num(current_string{2})*60 + str2num(current_string{3});
            else
                time(ii) = str2num(current_string{1})*60 + str2num(current_string{2});
            end
        end
        time = time - time(1) + 1;

        testvec = 1:1:time(end); testvec = testvec';
        
        for ii = 1:length(time)
            testvec(time(ii), 2) = (0.278*data.Var5(ii) + 0.075*data.Var5(ii))./bodymass;
        end
        
        for ii = 1:length(testvec)
            if testvec(ii,2) == 0
                testvec(ii,2) = testvec(ii-1,2);
            end
        end
        
        testvec(:,1) = testvec(:,1) - 1;

        metaData{subject}{session}{1} = delta;
        metaData{subject}{session}{2} = testvec(startidx:stopidx,2);
    end
end

save metabolicData.mat

%%



