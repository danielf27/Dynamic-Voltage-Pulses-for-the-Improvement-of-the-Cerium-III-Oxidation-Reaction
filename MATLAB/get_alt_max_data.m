% If any of the dimensions have a different range, input LB and UB as
% below:
% LB = [0.1 0.1 0.1];
% UB = [3 2 2];
% If all the dimensions have the same range, use single values as below:
LB = [100, 100, 1.1];
UB = [1000, 1000, 1.5];

noise = 0.1;
noise_str = '10';

date = '2021-06-01';

% dims = # of variable dimensions
dims = 3;

% init_num = # of initial data points to use
init_num = 10;

% batch_size = # of new experiments per batch of BO
batch_size = 3;

% total_num = total # of experiments to run before stopping
total_num = 50;

% num = number of points in each dimension.
% choose 40 for 2D space; 20 for 3D space; 15 for 4D space
% If you want a different number of points for each dimension, make
% different variables for each (e.g. num_1, num_2, etc.)
num = 20;

% Choose from MRB, PI, UCB, EI
acq_fun = 'MRB';

% Choose from BO, ChIDDO
BO_ChIDDO = 'ChIDDO';


%% Create comparison grid of design space
x_1 = linspace(LB(1),UB(1),num);
x_2 = linspace(LB(2),UB(2),num);
x_3 = linspace(LB(3), UB(3), num);
X_tot = zeros(num*num*num,dims);

for j = 1:length(x_1)
    for k = 1:length(x_2)
        for m = 1:length(x_3)
    % If adding another dimension, n would be:
            n = num*num*(j-1) + num*(k-1) + m;
%             n = num*(j-1) + k;
            X_tot(n, :) = [x_1(j) x_2(k) x_3(m)];
        end
    end
end

%% Define parameters/parameter ranges and create a set of alternative parameters
% params = the physics model parameters that is your estimated parameters
% of the system
params = [0.95, 1, 5e-8, 5e-10, 0.5, 0.5];
% param_std = estimated standard deviation of the possible parameter
% values. This is used to create alternate models for testing purposes.
param_std = [0.05, 0.05, 2e-8, 2e-10, 0.15, 0.15];

% Generate alternate parameter sets for testing purposes. 
alt_num = 20;
% alt_params = zeros(alt_num,length(params));
% for j = 1:length(params)
%     alt_params(:,j) = normrnd(params(j), param_std(j), [alt_num,1]);   
% end
% alts_filename = ['alts_' date '.csv'];
% writematrix(alt_params, alts_filename);

alt_name = ['alts_' date '.csv'];
alt_table = readtable(alt_name, 'ReadVariableNames', false);
alt_params = table2array(alt_table);
%% Calculate all the points of all the alternate models
% Calculate data for each of the alternate models
data = zeros(length(X_tot),alt_num);
for alt = 1:alt_num
    for row = 1:length(X_tot)
        data(row, alt) = electro_2(alt_params(alt,:), X_tot(row,:))  + 2*noise*rand - noise;
        row
        if data(row, alt) < 0
            data(row, alt) = 0.01;
        elseif data(row, alt) > 1
                data(row,alt) = 0.99;
        end
    end
end

%% Calculate the max value and location for each of alternate models
% Calculate max value for each of the alt models and save as csv
[alt_max, alt_inds] = max(data);
alt_max_data = cat(2,X_tot(alt_inds,:),alt_max.');
alt_filename = ['alt_max_data_' noise_str '_' date '.csv'];
writematrix(alt_max_data, alt_filename);