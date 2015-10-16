function TemperatureLogPlotter

% This script can be compiled with "mcc -e TemperatureLogPlotter.m" to
% produce a stand alone executable.
% File C:\Program Files\MATLAB\R2014b\bin\win64\tbb.dll should be placed in the
% same folder with the executable for the proper operatation.

[filename, pathname] = uigetfile({'*','Text Files';...
          '*.*','All Files' },'Select the ADR temperature log file',...
          'Z:\mcdermott-group\Data\ADR_log_files\');

if isnumeric(filename)
    return
end
      
if ~exist([pathname, filename], 'file')
    errordlg(['File ', pathname, filename, ' does not exist.'],...
        ['Error in ', mfilename]);
    return
end
      
data = readtable([pathname, filename], 'ReadVariableNames', false, 'Delimiter', '\t');
data = table2cell(data);
data(strcmp(data, '1.#QNAN0000000000000e+00')) = {NaN};
for k1 = 1:size(data, 1)
    for k2 = 1:size(data, 2)
        if ischar(data{k1, k2})
            data{k1, k2} = str2double(data{k1, k2});
        end
    end
end

if size(data, 2) ~= 5
    errordlg(['Data in file ', pathname, filename, ' cannot be recognized.'],...
        ['Error in ', mfilename]);
    return
end
if size(data, 1) < 2
        errordlg(['File ', pathname, filename, ' does not contain enough data.'],...
            ['Error in ', mfilename]);
    return
end

time = [data{:, 1}] / 60;
T_60K = [data{:, 2}];
T_03K = [data{:, 3}];
T_GGG = [data{:, 4}];
T_FAA = [data{:, 5}];

T_FAA(T_FAA >= 45 | T_FAA == 0) = NaN;
T_GGG(T_GGG >= 20 | T_GGG == 0) = NaN;

scrsz = get(0,'ScreenSize');
hght = scrsz(4);
figure('Name', 'ADR Temperature Log', 'Position',[.025*hght .25*hght .8*hght .65*hght])

plot(time, T_60K, 'b', time, T_03K, 'g', 'LineWidth', 2)
xlim([0 max(time)])
xlabel('Time (min)')
ylabel('Temperature (K)')
legend('60K Stage', '3K Stage')
title(['ADR Temperature Log [the log started at ',...
    filename(21:22), ':', filename(23:24), ' on ',...
     filename(16:17), '/', filename(18:19), '/', filename(14:15), ']'])

figure('Name', 'ADR Temperature Log', 'Position',[.85*hght .25*hght .8*hght .65*hght])
plot(time, T_03K, 'g', time, T_GGG, 'r', time, T_FAA, 'c', 'LineWidth', 2)
xlim([0 max(time)])
xlabel('Time (min)')
ylabel('Temperature (K)')
legend('3K Stage', '1K Stage (GGG)', '50mK Stage (FAA)') 
title(['ADR Temperature Log [the log started at ',...
    filename(21:22), ':', filename(23:24), ' on ',...
    filename(16:17), '/', filename(18:19), '/', filename(14:15), ']'])