%% MATLAB NN
% construct several neural networks using matlabs built in tools

% data readying
x=x_NN';
t=yield_data.production';

% number of nets to run
net_num=1;
predictions=zeros(4,size(t,2));
predictions_training_lm=zeros(size(t,2),size(t1,2));
predictions_training_gdx=zeros(size(t,2),size(t1,2));
predictions_training_br=zeros(size(t,2),size(t1,2));
errors=zeros(4,size(t,2));
errors_training_lm=zeros(size(t,2),size(t1,2));
errors_training_gdx=zeros(size(t,2),size(t1,2));
errors_training_br=zeros(size(t,2),size(t1,2));
training_record_lm=cell(size(t,2),1);
training_record_gdx=cell(size(t,2),1);
training_record_br=cell(size(t,2),1);

% split the data up to run hold one out validation
for l=1:size(t,2)
    index=1:size(t,2);
    index(l)=[];
    ind1=index;
    ind2=l;
    x1 = x(:,ind1);
    t1 = t(:,ind1);
    x2 = x(:,ind2);
    t2 = t(:,ind2);

    % feedforward net
    % currently training with bayesian weight and regularization scheme,
    % layout is 50,20,5, cascade seemed to work better
    tic
    net_ff_lm=feedforwardnet([30 15],'trainlm');
    net_ff_lm=configure(net_ff_lm,x1,t1);
    net_ff_lm=init(net_ff_lm);
    net_ff_lm.trainParam.showWindow=0;
    [net_ff_lm,tr_lm]=train(net_ff_lm,x1,t1);
    training_record_lm{l}=tr_lm;
    toc
 
    tic
    net_ff_gdx=feedforwardnet([30 15],'traingdx');
    net_ff_gdx=configure(net_ff_gdx,x1,t1);
    net_ff_gdx=init(net_ff_gdx);
    net_ff_gdx.trainParam.showWindow=0;
    [net_ff_gdx,tr_gdx]=train(net_ff_gdx,x1,t1);
    training_record_gdx{l}=tr_gdx;
    toc
    
    tic
    net_ff_br=feedforwardnet([30 15],'trainbr');
    net_ff_br=configure(net_ff_br,x1,t1);
    net_ff_br=init(net_ff_br);
    net_ff_br.trainParam.showWindow=0;
    [net_ff_br,tr_br]=train(net_ff_br,x1,t1);
    training_record_br{l}=tr_br;
    toc
    

    % cascade forward net
    % run 1 from fresno only gave bad results, 
    % currently training with bayesian weight and regularization scheme,
    % layout is 50,20,5
%     tic;
%     net_cf=cascadeforwardnet([30 15]);
%     net_cf=configure(net_cf,x1,t1);
%     net_cf=init(net_cf);
%     net_cf.trainParam.showWindow=0;
%     net_cf=train(net_cf,x1,t1);
%     toc;


    % find the predictions for each net for each input
    predictions(1,l)=net_ff_lm(x2);  
    %predictions(2,l)=net_cf(x2);  
    predictions(3,l)=net_ff_gdx(x2);
    predictions(4,l)=net_ff_br(x2);
    
    % find the predictions for each net on the training set
    predictions_training_lm(l,:)=net_ff_lm(x1); 
    predictions_training_gdx(l,:)=net_ff_gdx(x1);
    predictions_training_br(l,:)=net_ff_br(x1);
    
    % calculate the error
    errors(:,l)=100*abs(t2-predictions(:,l))./t2;
    errors_training_lm(l,:)=100*abs(t1-predictions_training_lm(l,:))./t1;
    errors_training_gdx(l,:)=100*abs(t1-predictions_training_gdx(l,:))./t1;
    errors_training_br(l,:)=100*abs(t1-predictions_training_br(l,:))./t1;
end

% find the mean error and accuracy
mean_error=mean(errors,2);
accuracy=100-errors;
mean_accuracy=mean(accuracy,2);