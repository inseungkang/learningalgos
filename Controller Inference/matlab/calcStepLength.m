% SLA calculation (based on right foot contact)

function SLA = calcStepLength(data, rightSpeed, leftSpeed)
    heelstrike_r = find(diff(data.Force_R(:,3) > 200) == 1);
    heelstrike_l = find(diff(data.Force_L(:,3) > 200) == 1);

    if length(heelstrike_r) > length(heelstrike_l)
        heelstrike_r = heelstrike_r(1:length(heelstrike_l));
    else
        heelstrike_l = heelstrike_l(1:length(heelstrike_r));
    end

    SLA = [];
    for ii = 1:length(heelstrike_r)
        steplength_r = data.Foot_R(heelstrike_r(ii),2) - data.Foot_L(heelstrike_r(ii),2);
        steplength_l = data.Foot_L(heelstrike_l(ii),2) - data.Foot_R(heelstrike_l(ii),2);
      
        SLA_idx = round((heelstrike_l(ii) + heelstrike_r(ii))/2);
        if rightSpeed >= leftSpeed
            SLA = [SLA; [SLA_idx, (steplength_r - steplength_l)/(steplength_r + steplength_l)]];
        elseif rightSpeed < leftSpeed
            SLA = [SLA; [SLA_idx, (steplength_l - steplength_r)/(steplength_r + steplength_l)]];
    end
end
