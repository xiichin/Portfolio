% tests K-SVD (OMP) with artificial experiment
% Brian Ng
% 10/08/2020

clear;
close all;

addpath(pwd);

cd tool/;
addpath(genpath(pwd));
cd ..;

cd plotter/;
addpath(genpath(pwd));
cd ..;

%% parameters
S = 5;      % sparsity
N = 20;     % length of each observation, i.e. time series vector, based on the journal 20*50
L = 100;    % number of observations
K = 50;     % dictionary size
maxIts = 120;
snr = 30;   % dB

%% Generate K-sparse training data, using random dictionary
W0 = zeros(K,L);                % Initialize zeros
D_gen = randn(N, K);            % Initialize random distributed number 
for ii = 1:L                    % Generate non-zero value randomly with sparsity of S = 3
    p = randperm(K);
    W0(p(1:S),ii) = randn(S,1);
end
nn = randn(N,L);
% add noise, SNR measured to average signal power
D_gen = D_gen./repmat(sqrt(sum(abs(D_gen).^2,1)),N,1);  % normalise columns norm 1
X = D_gen*W0;
P_s = mean(abs(X).^2,1);
nn = repmat(sqrt(10^(-snr/10)*P_s),N,1).*nn;

Y = X;% + nn;              % dafault Y = X+nn, try to remove the noise

%% Generate K-sparse training data and Y_test,unseen data 

W1 = zeros(K,L);
for jj = 1:L
    p1 = randperm(K);
    W1(p1(1:S),jj) = randn(S,1);
end
nn1 = randn(N,L);

X_test = D_gen*W1;
P_s1 = mean(abs(X_test).^2,1);
nn1 = repmat(sqrt(10^(-snr/10)*P_s1),N,1).*nn1;

Y_test = X_test + nn1;          % Create the unseen data with no noise

%% Run K-SVD
% initialise with random matrix

D0 = randn(N,K);                                % Generate educated guess or randomise of dictionary value
D0 = D0./repmat(sqrt(sum(abs(D0).^2,1)),N,1);   % normalise dictionary 
D = D0;
rmse = zeros(maxIts,1);
W = zeros(K,L);

ts = tic;
for it = 1:maxIts
    % solve pursuit (OMP)
    parfor ii = 1:L
        %% Apply OMP
        [~, ~, W(:,ii), ~, ~] = OMP(S, D, Y(:,ii), 1e-6);
    end
    % apply SVD to update the dictionary 
    R = Y - D*W;                                %  default R = Y - D*W;
    for k=1:K
        I = find(W(k,:));
        if ~isempty(I)
            % restricted error matrix
            Ri = R(:,I) + D(:,k)*W(k,I);        % default Ri = R(:,I) + D(:,k)*W(k,I);
            [U,Sigma,V] = svds(Ri,1);               
            D(:,k) = U;                         % default D(:,k) = U;
            W(k,I) = Sigma*V';
            R(:,I) = Ri - D(:,k)*W(k,I);        % default R(:,I) = Ri - D(:,k)*W(k,I);
        end
    end
    rmse(it) = sqrt(sum(abs(R(:).^2))/numel(R));
    if (rmse(it) < 1e-10)
        break;
    end
end
t1 = toc(ts);

         % store updated dictionary into D_trained
%% (2nd Part) Run the unseen output with the trained dictionary
rmse_test = zeros(maxIts,1);
W_test = zeros(K,L);

ts = tic;

for tt = 1:maxIts
    parfor ii = 1:L
        %% Apply OMP
        [~, ~, W_test(:,ii), ~, ~] = OMP(S, D, Y_test(:,ii), 1e-6);
    end
    % apply SVD to update the trained dictionary 
    R_test = Y_test - D*W_test;                                %  default R = Y - D*W;
    rmse_test(tt) = sqrt(sum(abs(R_test(:).^2))/numel(R_test));
    if (rmse_test(tt) < 1e-10)
         break;
    end    
end
t2 = toc(ts);
%% results
% compare distance between dictionaries
[D_dist, D_angle] = dictionary_distance(D,D_gen);          % default is dictionary_distance(D_true,D_gen)
E = X-D*W;
usage = zeros(K,1);

parfor k1 = 1:K
    usage(k1) = numel(find(W(k1,:)));
end
fprintf(1,'Trained dictionary with intial data\n');
fprintf(1,'Training time: %g s\n',t1);
%         fprintf(1,'# atoms recovered: %i\n',sum(diag(D_angle(:,am)) < beta_limit));
fprintf(1,'Rec error: %g\n',norm(E,'fro')/L);
fprintf(1,'Output SNR error: %g dB\n',20*log10(norm(Y,'fro')/norm(R,'fro')));

% compare distance between trained dictionary
[D_dist_trained, D_angle_trained] = dictionary_distance(D,D_gen);          % default is dictionary_distance(D_true,D_gen)
E_trained = X_test-D*W_test;
usage_trained = zeros(K,1);

parfor k_trained = 1:K
    usage_trained(k_trained) = numel(find(W_test(k_trained,:)));
end
fprintf(1,'Trained dictionary with unseen data\n');
fprintf(1,'Training time: %g s\n',t2);
%         fprintf(1,'# atoms recovered: %i\n',sum(diag(D_angle(:,am)) < beta_limit));
fprintf(1,'Rec error: %g\n',norm(E_trained,'fro')/L);
fprintf(1,'Output SNR error: %g dB\n',20*log10(norm(X_test,'fro')/norm(E_trained,'fro')));

% find average and std_dev for true dictionary 
parfor m = 1:L
    ss(m) = numel(find(abs(W(:,m)) > 0));
end

average = mean(ss);
std_dev = std(ss);

% find average and std_dev for trained dictionary 
parfor m = 1:L
    ss_trained(m) = numel(find(abs(W_test(:,m)) > 0));
end

average_trained = mean(ss_trained);
std_dev_trained = std(ss_trained);


%% Plots
figure(1);
h = figure(1);
plot(1:maxIts,rmse,'b','DisplayName','initial data','LineWidth',2); grid on;
hold on;
%plot(1:maxIts,rmse_test,'r','DisplayName','unseen data','LineWidth',2); grid on;
xlabel('Iteration'); ylabel('RMSE');
title('(RMSE vs iteration)');
saveas(h,['omp', '.png'], 'png');
legend;

% figure(2);
% plot(1:maxIts,rmse_test,'b'); grid on;
% xlabel('Iteration'); ylabel('RMSE');
% title('Unseen data (RMSE vs iteration)');

figure(3);
imagesc(D_angle); colorbar;
title('Atom angle separation (generated data)');

% figure(4);
% imagesc(D_angle_trained); colorbar;
% title('Atom angle separation (unseen data)');

figure(5);
imagesc(abs(W)>0);
title('Generated data (Sparse coefficient)');

figure(6);
imagesc(abs(W_test)>0);
title('Unseen data (Sparse coefficient)');

Table = table([S;S],[average;average_trained],[std_dev;std_dev_trained],'VariableNames',{'Sparsity','Average','Standard deviation'},'RowNames',{'Initial data','Unseen data'})