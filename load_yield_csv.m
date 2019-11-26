function [ yield_data ] = load_yield_csv( filepath )
%UNTITLED Load a csv formatted in two columns, year in the first column and
% production in the second

startRow = 2;
delimiter = ','; formatSpec = '%f%f%[^\n\r]';
fileID = fopen(filepath,'r');

dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
fclose(fileID);
yield_data=table(dataArray{:,1},dataArray{:,2},'VariableNames',{'year' 'production'});
clearvars filename delimiter formatSpec fileID dataArray ans files i;

% sort the yield values
yield_data=sortrows(yield_data,1);



end

