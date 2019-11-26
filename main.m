%% Import
% import the MODIS NDVI images, the yield values, and the chirps precip data
% load the MODIS tiffs
cd '/Users/noahdewar/Documents/Classes/CS 229/project'
clear

% load all the MODIS data in one go
cd ca_ag_ml/MCD43
files=dir('**/*.csv');
cd ../..
MODIS=cell(numel(files),1);
county_code=zeros(numel(files),1);
for i=1:numel(files)
    MODIS{i}=import_ndvi(strcat(files(i).folder,'/',files(i).name));
    county_code(i)=str2double(files(i).folder(end-2:end));
end

% drop san benito for now because it doesnt have trimm, drop inyo because
% its yield values are tiny, also drop madera
%county_codes=[string('107');string('037');string('019');string('029');string('031');string('027');string('069');string('053')];
%county_names=[string('tulare');string('madera');string('fresno');string('kern');string('kings');string('inyo');string('san_benito');string('monterey')];

% % only doing Tulare and Fresno for now
% county_codes=[string('107');string('019')];
% county_names=[string('tulare');string('fresno')];
% %county_codes=[string('107')];
% %county_names=[string('tulare')];
% 
% for k=1:numel(county_codes)
%     tic
%     eval(['cd ca_ag_ml/modis_data/' char(county_codes(k))]);
%     %eval(['cd ca_ag_ml/MCD43/' char(county_codes(k))]);
%     files=dir('*.csv');  
%     cd ../../..
%     eval(['MODIS_' char(county_names(k)) '=cell(1,numel(files));']);
%     for i=1:numel(files)
%         eval(['MODIS_' char(county_names(k)) '{i}=import_ndvi(strcat(files(i).folder,''/'',files(i).name));']);
%     end
%     toc
% end
% 
% % load the yield values for the counties being run
% for i=1:numel(county_names)
%     eval(['filename=char(''ca_ag_ml/' char(county_names(i)) '_y.csv'');'])
%     eval(['yield_data_' char(county_names(i)) '=load_yield_csv(filename);']);   
% end
% 
% % drop years before the MODIS data
% for i=1:numel(county_codes)   
%     eval(['yield_data_' char(county_names(i)) '(1:end-16,:)=[];']);
% end
% 
% % clean up
% clear filename files i k

% load all the TRIMM data in one go
cd ca_ag_ml/TRMM_3B42
files=dir('**/*.csv');
cd ../..
TRIMM=cell(numel(files),1);
county_code=zeros(numel(files),1);
for i=1:numel(files)
    TRIMM{i}=import_trimm(strcat(files(i).folder,'/',files(i).name));
    county_code(i)=str2double(files(i).folder(end-2:end));
end

% % load the trimm precip data
% for k=1:numel(county_codes)
%     eval(['cd ca_ag_ml/TRMM_3B42/' char(county_codes(k))]);
%     files=dir('*.csv');  
%     cd ../../..
%     eval(['TRIMM_' char(county_names(k)) '=cell(1,numel(files));']);
%     for i=1:numel(files)
%         eval(['TRIMM_' char(county_names(k)) '{i}=import_trimm(strcat(files(i).folder,''/'',files(i).name));']);
%     end
% end
% 
% % replace the vector with each locations precipatiton with the yearly
% % average
% for i=1:numel(county_names)
%     for j=1:numel(TRIMM_fresno)
%         eval(['TRIMM_' char(county_names(i)) '_yr_averages(j)=mean(TRIMM_' char(county_names(i)) '{j}(2:end));']);
%     end
% end

clear j k i files

%% Format the data
% set how many bins we will use for the NDVI and the bin width for the
% yield data
% these ara parameters that needs to be tuned
number_ndvi_bins=40;
bin_width_yield=300000;

% bin the ndvi data
for i=1:numel(county_codes)
    eval(['x_' char(county_names(i)) '=bin_modis_data(MODIS_' char(county_names(i)) ',number_ndvi_bins);']);
end

% stick the ndvi data together
for i=1:numel(county_codes)
    if i==1   
        eval(['x=x_' char(county_names(i)) ';']);
    else
        eval(['x=[x;x_' char(county_names(i)) '];']);
    end
end

% stick the TRIMM data together
for i=1:numel(county_codes)
    if i==1   
        eval(['TRIMM=TRIMM_' char(county_names(i)) '_yr_averages'';']);
    else
        eval(['TRIMM=[TRIMM;TRIMM_' char(county_names(i)) '_yr_averages''];']);
    end
end

% normalize the TRIMM data and then scale it be the mean of the x data
%TRIMM=(TRIMM-mean(TRIMM))/std(TRIMM);
TRIMM=5.5*TRIMM.*mean(mean(x,2));

% add it to the ndvi data
x_NB=[x TRIMM];
x_NN=[x TRIMM];

% stick yield tables together
for i=1:numel(county_codes)
    if i==1   
        eval(['yield_data=yield_data_' char(county_names(i)) ';']);
    else
        eval(['yield_data=[yield_data;yield_data_' char(county_names(i)) '];']);
    end
end

% find the range of values in the yield data
yield_values=min(yield_data.production):bin_width_yield:max(yield_data.production);

% make sure the last bin is long enough to capture the highest value
yield_values(end)=yield_values(end)+(max(yield_data.production)-max(yield_values))+10;

% sort the yield data into bins
y=zeros(length(yield_data.year),length(yield_values)-1);
for i=1:numel(yield_values)-1
    y(:,i)=double(yield_data.production>=yield_values(i))-double(yield_data.production>yield_values(i+1));
end

% clean up
clear i current_MODIS j x_temp x_means x chirps_total

%% Train/Test ML
% set the middle values of the bins to be used when calculating error
bin_middle_values=yield_values(1)+bin_width_yield/2:bin_width_yield:yield_values(end);

% see what the baseline error is for the binning of the yield values
bin_error=zeros(1,numel(y(:,1)));
for i=1:numel(y(:,1))
    bin_error(i)=100*(abs(bin_middle_values(logical(y(i,:)))-yield_data.production(i))/yield_data.production(i));
end

% mean error from binning is 0.66%, pretty low
mean_bin_error=mean(bin_error);

% train and run both Naive Bayes and the NN
NB=1;
NN=0;
NN_matlab=0;
plotting=0;
GSA=0;

%---------------------------------------------------------------
%===============================================================
%---------------------------------------------------------------
% train the Naive Bayes alg on all but one year and test it on the other
% years

% set up some vectors
predicted_yield_train=zeros(numel(y(1:size(x_NB,1),1)),31);
error_NB_train=zeros(numel(y(1:size(x_NB,1),1)),31);
if NB==1
    output_mat=zeros(numel(y(1:size(x_NB,1),1)),numel(y(1,:)));
    x_NB_current=x_NB;
    for i=1:numel(y(1:size(x_NB,1),1))
        index=1:numel(y(1:size(x_NB,1),1));
        index(i)=[];
        test_x=x_NB_current(i,:);
        test_y=y(i,:);
        trainMatrix=x_NB_current(index,:);
        trainy=y(index,:);
        [logyieldPhi,yield_prior]=train_nb(trainMatrix,trainy);
    
        % now test the trained parameters on the test dataset to get the test error
        output=test_nb(test_x,test_y,logyieldPhi,yield_prior);
        output_train=test_nb(trainMatrix,trainy,logyieldPhi,yield_prior);
    
        % convert the output to the middle bin value
        predicted_yield(i)=bin_middle_values(output>0);
        bin_middle_train=repmat(bin_middle_values,size(trainMatrix,1),1);
        predicted_yield_train(i,:)=bin_middle_train(output_train>0)';
    
        % add the trend back if needed
        output_mat(i,:)=output;
        error_NB(i)=abs(yield_data.production(i)-predicted_yield(i))/yield_data.production(i);
        error_NB_train(i,:)=abs(yield_data.production(index)'-predicted_yield_train(i,:))./yield_data.production(index)';
    end
    
    % make a vector with mean training error for each year
    mean_training_error_NB=mean(error_NB_train,2)';
    
    % make the figure
    figure(1)
    subplot(1,2,1)
    plot(2000:2000+numel(y(1:16,1))-1,100-error_NB(1:16)*100,'bx-')
    hold on
    plot(2000:2000+numel(y(1:16,1))-1,100-mean_training_error_NB(1:16)*100,'b.:')
    plot(2000:2000+numel(y(1:16,1))-1,100-error_NB(17:32)*100,'ro--')
    plot(2000:2000+numel(y(1:16,1))-1,100-mean_training_error_NB(17:32)*100,'r.:')
    legend('Location','southeast',{'Tulare Test' 'Tulare Training' 'Fresno Test' 'Fresno Training'});
    ylabel('Percent Accuracy')
    xlabel('Year Held out')   
    set(gcf,'pos',[100 100 900 900]);
    set(gca,'FontSize',16)
    grid on
    %figure(2)
    subplot(1,2,2)
    plot(2000:2000+numel(y(1:16,1))-1,predicted_yield(1:16),'bx:')
    hold on
    plot(2000:2000+numel(y(1:16,1))-1,predicted_yield(17:end),'ro:')
    plot(2000:2000+numel(y(1:16,1))-1,yield_data.production(1:16),'b-')
    plot(2000:2000+numel(y(1:16,1))-1,yield_data.production(17:end),'r-')
    legend({'Predicted Production Tulare' 'Predicted Production Fresno' 'True Production Tulare' 'True PRoduction Fresno'});
    ylabel('Production')
    xlabel('Year Held out')
    set(gca,'FontSize',16)
    set(gcf,'pos',[100 100 800 500]);
    ylim([5000000 35000000]);
    grid on
    
    % 26.1599% mean training error for NB across both counties
    % 73.8401% mean training accuracy for NB across both counties
    
    % 29.0550% mean test error for NB across both counties
    % 70.9450% mean test accuracy for NB across both counties

    % clean up from the NB testing and training
    clear i logyieldPhi index trainMatrix trainy test_x test_y
end

%---------------------------------------------------------------
%===============================================================
%---------------------------------------------------------------
% train a simple NN on the data

cd NN

% set up the GSA
if GSA==1
    monte_carlo_num=10000;
	learning_rate_vector=2+5*rand(1,100000);
	epoch_vector=randi([10 500],1,100000);
	lambda_vector=rand(1,100000).*10.^(-randi([2 5],1,100000));
	h1_vector=randi([10 500],1,100000);
	init_factor_vector=rand(1,100000).*10.^(-randi([1 3],1,100000));
    result=zeros(10000,9);
    %for kk=1032:monte_carlo_num
end
if NN==1

    % set up the vectors to hold the loss and accuracy
    %num_epoch = epoch_vector(kk);
    num_epoch = 100;
    mean_dev_accuracy=zeros(1,size(x_NN,1));
    mean_dev_loss=zeros(1,size(x_NN,1));
    mean_train_accuracy=zeros(1,size(x_NN,1));
    mean_train_loss=zeros(1,size(x_NN,1));
    final_training_yield=zeros(size(x_NN,1),size(x_NN,1)-1);
    final_dev_yield=zeros(1,size(x_NN,1));

    x_NN_current=x_NN;
    y_current=y;
    for l=1:numel(y_current(:,1))
        
        % set up the index to hold out one year of one of the counties for 
        % testing
        test_index=1:numel(y_current(:,1));
        test_index(l)=[];
        X_test=x_NN_current(l,:);
        y_test=y_current(l,:);
        X_dev=X_test;
        y_dev=y_test;
        
        % set up the training data using the hold one out index
        train_data=x_NN_current(test_index,:);
        y_train=y_current(test_index,:);
        X_train=train_data;
        
        % set the size of the feature space for the training data
        n=size(train_data,2);
        
        % set the number of classes for softmax
        c=size(y_train,2);
        
        % set the number of examples in the training set
        train_size=size(train_data,1);

        main_nn;
    end
    %result(kk,1:5)=[learning_rate_vector(kk) epoch_vector(kk) lambda_vector(kk) h1_vector(kk) init_factor_vector(kk)];
    %result(kk,6:9)=[mean(mean_train_loss,2)' mean(mean_dev_loss,2)' mean(mean_train_accuracy,2)' mean(mean_dev_accuracy,2)'];

    %end
    
    
end
cd ..

%---------------------------------------------------------------
%===============================================================
%---------------------------------------------------------------
% train a simple matlab NN
if NN_matlab==1
cd NN
matlab_nn3;
cd ..
end

%% Analysis
% run some analysis on the data collected from the Monte Carlo

% run PCA on the result to see what components account for most of the
% variance
%data_from_MC=result(1:1032,[1:5 9]);
%[coeff,score,latent,tsquared,explained]=pca(data_from_MC);
% didnt work correctly, need to revisit how to do this

% plot the result for lambda, learning rate, and h1 with loss as color
scatter3(data_from_MC(:,2),data_from_MC(:,1),result(1:1032,9),10,'filled');

% take the log of the init factor data column and the lambda data column
result_logged=result(1:6286,:);
result_logged(:,3)=log(result_logged(:,3));
result_logged(:,5)=log(result_logged(:,5));

% sort the result by accuracy on the dev set
[~,sort_index]=sort(result_logged(1:6286,7));
sorted_result=result_logged(sort_index,:);

figure(1)

for i=1:5
    [~,sort_index]=sort(result_logged(1:6286,7));
    sorted_result=result_logged(sort_index,:);
    subplot(5,2,1+2*(i-1))
    histogram(sorted_result(1:1000,i),35,'Normalization','pdf')
    hold on
    histogram(sorted_result(1001:end,i),35,'Normalization','pdf')
    if i==1
        title('Dev Loss Histograms')
    end
    if i==3
        ylabel('Normalized Counts','FontSize',20,'position',[-23 0.2]);
    end
    
    [~,sort_index]=sort(result_logged(1:6286,9));
    sorted_result=result_logged(sort_index,:);
    subplot(5,2,2+2*(i-1))
    histogram(sorted_result(1:1000,i),35,'Normalization','pdf')
    hold on
    histogram(sorted_result(1001:end,i),35,'Normalization','pdf')
    if i==1
        title('Dev Accuracy Histograms');
        xlabel('Learning Rate','FontSize',12);
    end
    if i==2
        xlabel('Number of Training Epochs','FontSize',12);
    end
    if i==3
        xlabel('Regularization Factor Lambda','FontSize',12);
    end
    if i==4
        xlabel('Number of Nuerons in Hidden Layer','FontSize',12);
    end
    if i==5
        xlabel('Weight Initilization Factor','FontSize',12);
    end
end

% make histograms comparing the top third to the rest
% histogram for regularization factor
figure(1)
histogram(log(sorted_result(1:1000,3)),35,'Normalization','pdf')
hold on
histogram(log(sorted_result(1001:end,3)),35,'Normalization','pdf')
% accuracy: shows smaller lambda is probably better
% loss: as above, smaller than 0.000182

% histogram for epoch number
figure(2)
histogram(sorted_result(1:1000,2),35,'Normalization','pdf')
hold on
histogram(sorted_result(1001:end,2),35,'Normalization','pdf')
% accuracy: shows epoch should be between 80 and 94
% loss: biggest difference is between 66 and 90 epochs

% histogram for learning rate
figure(3)
histogram(sorted_result(1:1000,1),35,'Normalization','pdf')
hold on
histogram(sorted_result(1001:end,1),35,'Normalization','pdf')
% accuracy: inconclusive
% loss: shows learning rate should be higher, between 4.8 and 7

% histogram for number of nodes
figure(4)
histogram(sorted_result(1:1000,4),35,'Normalization','pdf')
hold on
histogram(sorted_result(1001:end,4),35,'Normalization','pdf')
% accuracy: inconclusive, need to do grid search
% loss: inconclusive, but don't need to do about 260, loss gets worse

% histogram for init factor for weights
figure(5)
histogram(log(sorted_result(1:1000,5)),35,'Normalization','pdf')
hold on
histogram(log(sorted_result(1001:end,5)),35,'Normalization','pdf')
% accuracy: smaller is better
% loss: smaller than 0.00169

%% plotting NN results
% nn from class code
figure;
plot(1:num_epoch,dev_loss_all,'b')
hold on
plot(1:num_epoch,train_loss_all,'r')
plot(1:num_epoch,mean(dev_loss_all),'k','LineWidth',2)
xlabel('Epochs');
ylabel('Loss');
set(gca,'FontSize',16)
set(gcf,'pos',[100 100 900 900]);
grid on

figure;
plot(1:num_epoch,100-train_accuracy_all,'r')
hold on
plot(1:num_epoch,100-dev_accuracy_all,'b')
plot(1:num_epoch,100-mean(dev_accuracy_all),'k','LineWidth',2)
xlabel('Epochs');
ylabel('Percent Accuracy');
set(gca,'FontSize',16)
set(gcf,'pos',[100 100 900 900]);
grid on

figure;
plot(2000:2015,final_dev_yield(1:16),'bx-')
hold on
plot(2000:2015,final_dev_yield(17:32),'ro-')
plot(2000:2015,yield_data.production(1:16),'kx-')
plot(2000:2015,yield_data.production(17:32),'ko-')
legend({'Predicted Yields Tulare' 'Predicted Yields Fresno' 'True Yields Tulare' 'True Yields Fresno'});
ylabel('Yield')
xlabel('Year Held out')
set(gca,'FontSize',16)
set(gcf,'pos',[100 100 900 900]);
grid on

%% plotting NN results
% nn from matlab toolbox
mean_training_accuracy=100-mean(errors_training_br,2)';
figure;
subplot(1,2,1)
plot(2000:2015,accuracy(4,1:16),'bx-')
hold on
plot(2000:2015,mean_training_accuracy(1:16),'b.:')
plot(2000:2015,accuracy(4,17:32),'ro--')
plot(2000:2015,mean_training_accuracy(17:end),'r.:')
legend({'Tulare Test' 'Tulare Training' 'Fresno Test' 'Fresno Training'},'location','southeast');
%legend({'Levenberg-Marquardt backpropagation Tulare' 'Fresno' 'Variable Learning Rate Gradient Descent Tulare' 'Fresno' 'Bayesian Regularization Tulare' 'Fresno'},'location','southeast');
xlabel('Year Held out')
ylabel('Percent Accuracy');
set(gca,'FontSize',16)
set(gcf,'pos',[100 100 900 900]);
grid on

%figure;
subplot(1,2,2)
plot(2000:2015,predictions(4,1:16),'bx:')
hold on
plot(2000:2015,predictions(4,17:end),'ro:')
plot(2000:2015,yield_data.production(1:16),'b-')
plot(2000:2015,yield_data.production(17:end),'r-')
legend({'Predicted Production Tulare' 'Predicted Production Fresno' 'True Production Tulare' 'True Production Fresno'});
ylabel('Production')
xlabel('Year Held out')
ylim([5000000 35000000]);
set(gca,'FontSize',16)
set(gcf,'pos',[100 100 800 500]);
grid on