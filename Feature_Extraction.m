                                                                     
                                                                              

%Matlab code for Home work Assignment-3  Part 2
% INSTRUCTIONS TO RUN THE CODE
% 1.Load the excel files of TRAIN_SET & TEST_SET into the current folder.
% 2.The same data set which was used in the first part of the assignment
% 3.It is not written in the function format and doesn't require any input
% arguments as the first part of the assignment.
%4. Run the code by clicking the run command. It takes a bit longer than
%first part of the assignment to complete execution
 
% CODE STARTS HERE
 
%Randomizing the data and generating test set and training set STARTED
clc
clear all
train_set=xlsread('TRAIN_SET');
test_set=xlsread('TEST_SET');
%Randomizing the data and generating test set and training set COMPLETED
 
%Defining number of Hidden layers and no.of neurons in hidden layer STARTED
HL=1; % No. of hidden layers
HN=[150]; % No. of neurons in hidden layers
%Defining number of Hidden layers and no.of neurons in hidden layer Complet
TN=[784 HN 784]; % TOTAL NUMBER OF NEURONS IN ALL LAYERS 
 
Weight_Matrices=cell(HL+1,1);
 
 
for i=1:1:HL+1
% Initializing both weight matrices STARTED
Weight_Matrices{i}=normrnd(0,sqrt(2/TN(i+1)),[TN(i+1) TN(i)+1]);
% Initializing both weight matrices COMPLETED
end
 
% Initializing both Activation function that is used STARTED
Yj = @(x) 1/(1+exp(-x));
% Initializing both Activation function that is used COMPLETED
 
% Initializing learning rate and momentum coefficient STARTED
ita_1=0.7;
alpha = 0.9;
Average_Activation_req = 0.05;
beta = 1; decay_constant = 0.00001;
% Initializing learning rate and momentum coefficient COMPLETED
 
% ALL PARAMETERS ARE DEFINED BY THIS POINT NO ERRORS ARE OBSERVED UNTIL
% THIS POINT. WE CREATED TRAINING SET AND TEST SET FROM RANDOMIZED DATA
% SAMPLE. DEFINED ALL THE REQUIRED COEFFICIENTS AND FUNCTIONS.
 
% LEARNING STARTS FROM THIS POINT. TRAINING DATA IS INPUT FROM THIS POINT
 
% AT THIS POINT INPUT TRAVELS FORWARD
J_Combined = [];
J_digits_train=zeros(1,10);
J_digits_test=zeros(1,10);
% We need to generate 150 outputs of hidden layer for 1 neuron
%define delta Wij delta Wjk initial matrices here.

Activated_Outputs=cell(HL+1,1);
der_Activated_Outputs=cell(HL+1,1);
J_mat=[];
% DEFINE NUMBER OF EPOCHS HERE
for epoch= 1:1:1500
    J_epoch=[];Total_Activation = zeros(HN,1);Activated_Outputs_Outputlayer = cell(4000,1);
    der_Activated_Outputs_Outputlayer = cell(4000,1);
    Activated_Outputs_Hiddenlayer = cell(4000,1);
    der_Activated_Outputs_Hiddenlayer = cell(4000,1);
    delta_Outputlayer = cell(4000,1);
    delta_Hiddenlayer = cell(4000,1);
% 4000 inputs go into hidden layer here
% Calculation of AVERAGE ACTIVATION & Error for all inputs
for inp=1:1:4000
Hjk = train_set(inp,1:784)';
for i=1:1:HL+1
     s=Weight_Matrices{i}*[1;Hjk];%here [1 TRAINSET] IS FOR NUMBER OF INPUT and 1 outside is for bias
    Hjk = arrayfun(Yj,s);
    Activated_Outputs{i}=Hjk;
    der_Activated_Outputs{i} = (1-Hjk).*Hjk;
 end
  Total_Activation = Total_Activation + Activated_Outputs{1,1};
  Activated_Outputs_Outputlayer{inp} = Activated_Outputs{2,1};
  der_Activated_Outputs_Outputlayer{inp} = der_Activated_Outputs{2,1};
  Activated_Outputs_Hiddenlayer{inp} = Activated_Outputs{1,1};
  der_Activated_Outputs_Hiddenlayer{inp} = der_Activated_Outputs{1,1};
  Yij = Activated_Outputs{HL+1};
  J_q_1 = (train_set(inp,1:784)'-Yij);
  J_q = (0.5*((J_q_1).^2));
  Jq = (sum(J_q));
  J_epoch(end+1)=Jq;
   if epoch==1000
    J_digits_train(train_set(inp,785)+1)=J_digits_train(train_set(inp,785)+1)+Jq;
  end
end
Average_Activation_Calculated = (Total_Activation)./4000;

% Average activation for the epoch is calculated 

% Terms of Kullback leibler divergence are defined here
A1 = (Average_Activation_req)./(Average_Activation_Calculated);
B1 = (1-Average_Activation_req)./(1-Average_Activation_Calculated);
%These values remain constant over an epoch and change for every new epoch

% Error is calculated. The terms of sparsness and weight decay must be added
if rem(epoch,10)==0 || epoch ==1
J_mat(end+1)= sum(J_epoch);
end
% Add error terms with appropriate formula. There is beta term here while
% adding and also all weight parameters must be squared and added. Add
% carefully
accum_del_Weights_Outputlayer = zeros(784,151);
del_Weights_Outputlayer = zeros(784,151);
for inp=1:1:4000
% calculating delta values
delta_Outputs = [train_set(inp,1:784)'-Activated_Outputs_Outputlayer{inp}].*der_Activated_Outputs_Outputlayer{inp};
delta_Outputlayer{inp} = delta_Outputs;
del_Weights_Outputlayer = (ita_1.*(delta_Outputs))*[1; Activated_Outputs_Hiddenlayer{inp}]' + (alpha.*del_Weights_Outputlayer);
accum_del_Weights_Outputlayer = accum_del_Weights_Outputlayer + del_Weights_Outputlayer;
end
%accum_del_Weights_Outputlayer = (accum_del_Weights_Outputlayer) - (0.4)*[zeros(784,1) Weight_Matrices{2,1}(:,2:TN(2)+1)];
% Wij change is completed here for 1 input

 
% Calculation of Wjk starts here. 
% Calculate del_Wjk
accum_del_Weights_Hiddenlayer = zeros(150,785);
del_Weights_Hiddenlayer = zeros(150,785);
for inp = 1:1:4000
delta_Inputs = ((Weight_Matrices{2,1}(:,2:TN(2)+1)'*(delta_Outputlayer{inp})) - beta*(B1-A1)).*(der_Activated_Outputs_Hiddenlayer{inp});
delta_Hiddenlayer{inp} = delta_Inputs; 
del_Weights_Hiddenlayer = [ita_1.*(delta_Inputs)]*[1 train_set(inp,1:784)] + [alpha.*(del_Weights_Hiddenlayer)];
accum_del_Weights_Hiddenlayer = accum_del_Weights_Hiddenlayer + del_Weights_Hiddenlayer;
end
%accum_del_Weights_Hiddenlayer = (accum_del_Weights_Hiddenlayer) - 4.*[zeros(150,1) Weight_Matrices{1,1}(:,2:TN(1)+1)];
 % Cumulative weights calculated. 
 
 %Updating weights
Weight_Matrices{1,1} = Weight_Matrices{1,1} + (accum_del_Weights_Hiddenlayer)./4000 - 4000*decay_constant*[zeros(150,1) Weight_Matrices{1,1}(:,2:TN(1)+1)];
Weight_Matrices{2,1} = Weight_Matrices{2,1} + (accum_del_Weights_Outputlayer)./4000 - 4000*decay_constant*[zeros(784,1) Weight_Matrices{2,1}(:,2:TN(2)+1)];
% Calculation of training error and validation error . SEPERATE SECTION
% UNTIL THIS POINT NO ERRORS. CHECK FROM THIS POINT
 
end




 
 
% Training is completed here
% Testing starts here
Activated_Outputs_test = cell(HL+1,1);
der_Activated_Outputs_test = cell(HL+1,1);
J_epoch_test=[];Total_Activation_test = zeros(HN,1);
for inp=1:1:1000
% ALWAYS VERIFY THIS CALCULATION PART IF ANSWER GOES WRONG
%The calculations below are for 1 neuron
% FROM INPUT LAYER TO HIDDEN LAYER
Hjk = [];F_der_sj=[]; % to store output values of hidden layer
Hjk = test_set(inp,1:784)';
for i=1:1:HL+1
 
    s=Weight_Matrices{i}*[1;Hjk];%here [1 TRAINSET] IS FOR NUMBER OF INPUT and 1 outside is for bias
    Hjk = arrayfun(Yj,s);
    Activated_Outputs_test{i}=Hjk;
    der_Activated_Outputs_test{i} = (1-Hjk).*Hjk;
    Total_Activation_test = Total_Activation_test + Activated_Outputs_test{1};
end
Yij =  Activated_Outputs_test{HL+1};
Average_Activation_test = (Total_Activation_test)./1000;

J_q_1_test = (test_set(inp,1:784)'-Yij);
J_q_test = (0.5*((J_q_1_test).^2));
J_test = (sum(J_q_test));
J_epoch_test(end+1)=J_test;
J_digits_test(test_set(inp,785)+1)=J_digits_test(test_set(inp,785)+1)+J_test; % eRROR FOR EACH DIGIT IS STORED HERE
 
 
end
  
J_mat_test=sum(J_epoch_test); % ERROR FOR ALL THE INPUTS IS COLLECTED HERE
    
 
% Plotting of Features. (Plotting the weight matrices
U = (Weight_Matrices{1}(:,2:end));
 
for i=1:10
    for j = 1:10
        v = reshape(U((i-1)*10+j,:),28,28);
        subplot(10,10,(i-1)*10+j)
        image(64*v)
        colormap(gray(64));
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])
        set(gca,'dataaspectratio',[1 1 1]);
    end
end 
% Plotting of the error function per digit
for i=1:1:10
J_Combined = [J_Combined;(J_digits_train(i))/4000 (J_digits_test(i))/1000];
end
c=categorical({'0','1','2','3','4','5','6','7','8','9'});
figure(2)
B=bar(c,J_Combined(1:10,:));
bar_legend=cell(1,2);
bar_legend{1}='Training Set';
bar_legend{2}='Test Set';
legend(B,bar_legend);
ylabel('Error Function');
xlabel('Digits');
title('Error Function per digit');
% Plotting error function over total number of epochs        
 figure(4) 
 plot([1 10:10:1500],J_mat);
 xlabel('epoch');
 ylabel('Error Function');
 title('Error Function vs Epoch');
c1 = categorical({'Train Set','Test Set'}); 
% Plotting final J values for test set and training set
figure(5)
B1 = bar(c1,[J_mat(end)/4000;(J_mat_test)/1000]);
bar_legend=cell(1,2);
bar_legend{1}='Training Set';
bar_legend{2}='Test Set';
ylabel('Error Function');
xlabel('Data Set');
title('Error Function vs Data Set');

