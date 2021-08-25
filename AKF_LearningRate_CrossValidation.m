% Ramesh Perumal: Sep 16, 2020
clear all;
close all;
clc;
% datasets = {'R2_M1D','R2_M1U','R1_M2D','R1_M2U','R3_SD','R1_SU','R3_STRI','R1_THAL'};
datasets = {'R1_M1D','R1_M1U','R1_M2D','R1_M2U','R1_SD','R1_SU','R1_STRI','R1_THAL','R2_M1D','R2_M1U','R2_STRI','R2_SD','R3_M1D','R3_M1U','R3_STRI','R2_SD','R4_M1D','R4_M1U','R4_SD','R4_SU'};
% R0 = [1.5e-4,3.1e-4,2.7e-4,7.2e-5,1.6e-4,1.6e-4,2.7e-4,2.3e-4];
R0 = [6.3e-5,1e-4,2.7e-4,7.2e-5,1.1e-4,1.6e-4,4e-4,2.3e-4,1.5e-4,3.1e-4,3.9e-4,1.6e-4,1.3e-4,1.1e-4,2.7e-4,1.6e-4,1.3e-4,7.5e-5,1.3e-4,1.15e-4];
data_len = length(datasets);
N = 60000;
data = zeros(N,data_len);
for i = 1:data_len
    data_loc = strcat('D:\Research\HVS\SignalProcessing\TimePointPrediction\UAAR\Code\Single_Channel_LFP_Prediction\AdaptivePrediction\Code\hvs_bp\preprocessed\val\',datasets{i},'.csv');
    tmp = importdata(data_loc);    
    data(:,i) = tmp(:,1);        
end
xn = data;
clear tmp;

%% Initialization
p = 6; % TVAR model order 
T = 24; % Modeling at Interval 
p_eff = p*T; % Effective model order
temp = zeros(p,1);
uc = [0,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3:1e-3:9e-3,1e-2:1e-2:10e-2,0.2:0.1:1];
len_uc = length(uc);
REV = zeros(data_len,len_uc);
model_fit = REV;
I = eye(p);
for j = 1:data_len
    fprintf('Dataset %d\n',j)
    norm_x = sqrt(sum(xn(p_eff+1:end,j).^2));
    var_x = var(xn(p_eff+1:end,j));
    for i = 1:len_uc
        fprintf('UC is %d\n',uc(i));
        X = temp; % px1 Input signal vector
        W = X; % px1 AR coefficient vector 
        S = I; % State estimation error covariance matrix 
        res = zeros(1,N); % Measurement residue

        % Adaptive Kalman filter parameters
        R = R0(j); % Without filtering        
        cov_Q = 0.1*R*I; % Process noise variance
        temp6 = I*(uc(i)/p);
        
        % Training Phase to determine the optimal AR coeffcients
        for n = p_eff+1:N         
            X(:,1) = xn(n-T:-T:n-p_eff,j); 
            % Prediction step of AKF
            x_est = W'*X; % Apriori estimate of x(n) at 'n'
            e = xn(n,j) - x_est; % Prediction Error
            S = S + cov_Q; % Apriori estimate of S    
            if uc(i) > 0
                % Adaptive estimation of R
                R = (1-uc(i))*R + uc(i)*e*e;
            end
            SX = S*X;
            temp2 = X'*SX + R; % Residual state error covariance matrix
            % Correction step of AKF
            K = SX./temp2; % Update Kalman gain K = (S*H')./(H*S*H'+R)    
            W = W + K*e; % Update W (Aposteriori value of W)    
            S = S-K*SX'; % Update S (Aposteriori value of S)    
            % Residue 
            res(n) = e;
            if uc(i) > 0
                cov_Q = temp6*trace(S);
            end        
        end        
        % Compute the Relative Error Variance (REV)
        mse = mean(res(p_eff+1:end).^2); % Mean Squared Error
        REV(j,i) = mse / var_x ;        
        fprintf('REV is %0.4f\n',REV(j,i));

        % Goodness-of-fit measure
        norm_res = sqrt(sum(res(p_eff+1:end).^2));        
        model_fit(j,i) = norm_res / norm_x;
        fprintf('Goodness-of-fit is %0.4f\n',model_fit(j,i));        
    end
end

%% Display the values of REV and model_fit for UC values till 0.01 across the 
% HVS datasets
% markers = {'-ks','-kd','-kv','-k^','-ko','-k*','-k<','-k>'}; 
% dataset_names = {'R2-M1D','R2-M1U','R1-M2D','R1-M2U','R3-SD','R1-SU','R3-STRI','R1-THAL'};

markers = {'-rs','-rd','-rv','-r^','-ro','-r*','-r<','-r>','-bs','-bd','-bv','-b^','-gs','-gd','-gv','-g^','-ks','-kd','-kv','-k^'}; 
dataset_names = {'R1-M1D','R1-M1U','R1-M2D','R1-M2U','R1-SD','R1-SU','R1-STRI','R1-THAL','R2-M1D','R2-M1U','R2-STRI','R2-SD','R3-M1D','R3-M1U','R3-STRI','R3-SD','R4-M1D','R4-M1U','R4-SD','R4-SU'};

f1 = figure('color',[1 1 1]);
uc_st = 2;
uc_end = 11;
% for j = 1:data_len        
%     figure(f1);subplot(1,2,1);
%     semilogx(uc(uc_st:uc_end),REV(j,uc_st:uc_end),markers{1,j},'MarkerEdgeColor','k','MarkerSize',8,'Linewidth',1);
%     hold on;    
% end
% xlabel('Learning Rate','FontSize',12,'Fontweight','b');
% ylabel('Relative Error Variance','FontSize',12,'Fontweight','b');
% legend(dataset_names);
% xlim([1e-8,1e-2]);
% ax1 = gca;
% ax1.XTick = uc([uc_st:7,uc_end]);
% ax1.FontSize = 12;
% ax1.FontWeight = 'b';

for j = 1:data_len        
    figure(f1);subplot(1,2,2);    
    semilogx(uc(uc_st:uc_end),model_fit(j,uc_st:uc_end),markers{1,j},'MarkerSize',5,'Linewidth',1);
    hold on;    
end
xlabel('Learning Rate','FontSize',12,'Fontweight','b');
ylabel('Normalized Residual','FontSize',12,'Fontweight','b');
legend(dataset_names);
xlim([1e-8,20e-2]);
ax2 = gca;
ax2.XTick = uc([uc_st:7,uc_end]);
ax2.FontSize = 12;
ax2.FontWeight = 'b';
ax2.Legend.FontSize = 10;
ax2.Legend.FontWeight = 'b';

