function [ fixed_csv ] = format_modis_csv( csv )
%UNTITLED3 format the cell arrays made from the csvs a little bit
current_rows=zeros(((numel(csv{1}(:,1))-1)/7),1);
fixed_csv=cell(1,numel(csv));
    for i=1:numel(csv)
        csv{i}(1,:)=[];
        for j=1:(numel(csv{i}(:,1))/7)
            current_rows(j,1)=mean(csv{i}(7*j-6:7*j,1));
        end
        fixed_csv{i}=current_rows;
        
    end



end

