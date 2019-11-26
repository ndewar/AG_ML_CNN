%% Format the Yield Data
% load all the csvs containing yield data for all of Cali's counties and
% then format it properly.
clear
cd csvs
files=dir('*.csv');

for l=1:length(files)
    % load the csvs in one at a time
    current_ca=import_yield_csv(files(l).name,2);

    % drop crops we don't care about
    bad_crops={'BEES' 'OSTRICH' 'SERVICE' 'HORSE' 'PASTURE' 'POULTRY' 'MILK' 'TURKEY' 'WOOL' 'APIARY' 'RABBITS' 'SHEEP' 'BIOMASS' 'CATTLE' 'CHICKEN' 'TREE' 'EGGS' 'FISH' 'FLOWER' 'FOREST' 'BIRD' 'GOAT' 'GAME' 'HOGS' 'PIG' 'LAMB' 'LIVESTOCK' 'MANURE' 'NURSERY'}';
    % remove SILAGE and FEED? not sure
    
    bad_index=zeros(numel(current_ca.CropName),1);
    for i=1:numel(bad_crops)
        temp_index=regexp(current_ca.CropName,bad_crops{i});
        for j=1:numel(temp_index)
            if isempty(temp_index{j})
                temp_index{j}=0;
            end
        end
        bad_index=bad_index+cell2mat(temp_index);
    end

    % fix the retrieved indices from regexp
    bad_index(bad_index>1)=1;
    bad_index=1-bad_index;

    % make a smaller table without the crops we want to drop
    current_ca_trim=current_ca(logical(bad_index),:);

    % find out which crops have nan in production
    nan_prod=current_ca_trim(isnan(current_ca_trim.Production),:);

    % make a table of just the state totals
    temp_index=strcmp(current_ca.County,'State Totals');
    current_ca_totals=current_ca(temp_index,:);

    for i=1:numel(nan_prod.Value)
        temp_index=strcmp(current_ca_totals.CropName,nan_prod.CropName{i});
        
        % replace the crops that have nans with their proportion of the
        % total production from the state total based on the proportion of
        % their value to the total value
        nan_prod.Production(i)=current_ca_totals.Production(temp_index)*nan_prod.Value(i)/current_ca_totals.Value(temp_index);
        clear temp_index
    end
    
    % replace the nan values in our trimmed table with the values
    % calculated
    current_ca_trim.Production(isnan(current_ca_trim.Production),:)=nan_prod.Production;
    if l==1
        y_table=current_ca_trim;
    else
        y_table=[y_table;current_ca_trim];
    end

end
cd ..

% remove empty crop types
y_table(y_table.CommodityCode<100000,:)=[];

% get just Tulare
years=1980:2015;
Tulare=[y_table(strcmp(y_table.County,'Tulare '),:);y_table(strcmp(y_table.County,'Tulare'),:)];
Tulare(isnan(Tulare.Production),:)=[];
Tulare_Y=zeros(numel(years),2);
for i=1:numel(years)
    Tulare_Y(i,1)=years(i);
    Tulare_Y(i,2)=sum(Tulare.Production(Tulare.Year==years(i)));
end

% write the table
writetable(y_table);

% write the Tulare table
Tulare_y_tab=table(Tulare_Y(:,1),Tulare_Y(:,2),'VariableNames',{'year' 'production'});
writetable(Tulare_y_tab,'Tulare_Y.csv') 
type Tulare_Y.csv;
