function [ x ] = bin_modis_data( modis_ndvi,number_ndvi_bins)
%UNTITLED5 bin the ndvi modis data, input should be a cell array with one
% long column vector per cell representing a year of modis ndvi data
x=zeros(numel(modis_ndvi),number_ndvi_bins);
for i=1:numel(modis_ndvi)
    
    % make the MODIS_column into row vector with 20 features, each is
    % simply counting the number of times an ndvi value in that range
    % appeared
    current_MODIS=modis_ndvi{i};
    x_current=zeros(1,number_ndvi_bins);
    for j=1:number_ndvi_bins
        x_current(j)=sum(double(current_MODIS<j/number_ndvi_bins)-double(current_MODIS<=(j-1)/number_ndvi_bins));
    end
    x(i,:)=x_current;
    
end

end

