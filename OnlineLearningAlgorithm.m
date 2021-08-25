%% Online-Learning Algorithm using Adaptive Kalman Filter (AKF)
% This script implements the online-learning algorithm comprising the
% time-varying autoregressive (AR) model at interval whose coefficients 
% are updated by AKF. The power spectral denisty (PSD) is estimated from the 
% model coefficients, and the HVS power is obtained by the sum of PSD over 
% 5-13 Hz
% 
% Programmed by Ramesh Perumal, March 18, 2020
%%
clear all;
close all;
clc;
% Load the LFP data
data = importdata('Preprocessed\val\R1_M1D.csv');
xp = data(:,1); % Preprocessed LFP

% Preprocessing: Uncomment the lines 20-21 to preprocess the raw LFP data.
% The value of alpha (truncation factor) is determined from Table 3.2 in  
% the thesis 
% alpha = 0.001;
% xp = preprocess(xp,alpha);
st_time = 1; % Starting timestamp in LFP recordings
N = length(xp);

%% Initialization
p = 6; % AR model order 
T = 24; % Modeling interval 
p_eff = p*T; % Effective model order
X = zeros(p,1); % px1 Input signal vector
W = X; % px1 TVAR coefficient vector (or state vector)
e = zeros(N,1); % Measurement residuals  
S = eye(p); % State estimation error covariance matrix 
x_est = zeros(N,1); % Measurement estimate 

% Adaptive Kalman filter parameters 
% R0 = var(diff(xn(t1:t2))); % After normalization,but without filtering
r = 6.25e-5; % Initial value of measurement noise variance
Q = 0.1*r*eye(p); % Process noise covariance matrix
uc = 0.005; % Optimized learning rate of AKF
temp6 = eye(p)*(uc/p);
hvs_bp = zeros(N,1); % HVS Bandpower
tr = 0.15; % Optimized detection threshold for R1_M1D
hvs = hvs_bp; % Detection Response

% Parameters for estimating PSD from the AR model coefficients 
b = 1;
fs = 42; % Sampling frequency in Hz
Nf = 256; % No of points for computing frequency response
% Frequency indices corresponding to the 5-13 Hz band 
lf = 62; hf = 160;
psd = zeros(1,Nf);

%% Algorithm
tic;
for n = p_eff+1:N
    % Down-sampling the LFP segment from 1 kHz to 42 Hz
    X(:,1) = xp(n-T:-T:n-p_eff,1); 
    % Prediction step of AKF
    x_est(n) = W'*X; % A priori estimate of xp(n) at 'n'
    e(n) = xp(n) - x_est(n); % Prediction Error    
    S = S + Q; % Apriori estimate of S
    % Adaptive estimation of R
    r = (1-uc)*r + uc*e(n)*e(n);
    SX = S*X;
    temp2 = X'*SX + r; % Innovation covariance matrix
    % Correction step of AKF
    K = SX./temp2; % Update Kalman gain K = (S*H')./(H*S*H'+R)    
    W = W + K*e(n); % Update W (A posteriori value of W)    
    S = S-(K*SX'); % Update S (A posteriori value of S) 
    % Adaptive estimation of Q
    Q = temp6*trace(S);    
    % Estimating PSD from the model coefficients     
    a = [1,-1*W'];        
    % Frequency Response of the AR Coefficients
    [h,fout] = freqz(b,a,Nf,fs);
    psd = r*(abs(h).^2);
    hvs_bp(n) = sum(psd(lf:hf));    
    if hvs_bp(n) > tr
        hvs(n) = 1;
    end    
end
toc;

%% Display the detection response
figure('color',[1 1 1]);
subplot(2,1,1);
plot(p_eff+1:N,xp(p_eff+1:N),'r','Linewidth',1);
hold on;
plot(p_eff+1:N,hvs(p_eff+1:N),'k','Linewidth',1);
ylim([-0.2,1.2]);

subplot(2,1,2);
plot(p_eff+1:N,hvs_bp(p_eff+1:N),'k','Linewidth',1);
xlabel('Time (ms)','FontSize',12,'Fontweight','b');
ylabel('HVS Band Power','FontSize',12,'Fontweight','b');
