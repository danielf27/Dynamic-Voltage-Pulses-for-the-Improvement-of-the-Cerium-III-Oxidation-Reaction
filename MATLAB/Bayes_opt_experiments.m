%% Bayesian Optimization for experiments (MATLAB)

%% Define hyperparameters
% If any of the dimensions have a different range, input LB and UB as
% below:
% LB = [0.1 0.1 0.1];
% UB = [3 2 2];
% If all the dimensions have the same range, use single values as below:
LB = [5, 5];
UB = [200, 200];

date = '2022-04-06';

% dims = # of variable dimensions
dims = length(LB);

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
num = 40;

% Choose from MRB, PI, UCB, EI
acq_fun = 'MRB';

% Choose from BO, ChIDDO
BO_ChIDDO = 'BO';

 
%% Create comparison grid of design space
x_1 = linspace(LB(1),UB(1),num);
x_2 = linspace(LB(2),UB(2),num);
% x_3 = linspace(LB(3), UB(3), num);
X_tot = zeros(num*num,dims);

for j = 1:length(x_1)
    for k = 1:length(x_2)
%         for m = 1:length(x_3)
    % If adding another dimension, n would be:
%             n = num*num*(j-1) + num*(k-1) + m;
        n = num*(j-1) + k;
        X_tot(n, :) = [x_1(j) x_2(k)];
    end
end

%% Define parameters/parameter ranges and create a set of alternative parameters
% params = the physics model parameters that is your estimated parameters
% of the system
params = [1e-7, 1e-9, 5e-11];
% param_std = estimated standard deviation of the possible parameter
% values. This is used to create alternate models for testing purposes.
param_std = [0.05, 0.05, 2e-8, 2e-10, 0.15, 0.15];

% Generate alternate parameter sets for testing purposes.

% %% Select initial experiments
% if length(UB)==1
%     init_points = (UB-LB).*rand(init_num,dims) + LB;
%     % If you're variable only goes to a certain decimal point, use the
%     % round function below:
%     init_points = round(init_points, 1);
% else
%     init_points = zeros(init_num, dims);
%     for j = 1:length(UB)
%         init_points(:,j) = (UB(j)-LB(j)).*rand(init_num,1) + LB(j);
%     end
% end
% 
% %% Save the initial points in a file
% init_filename = ['Init_points_electro_3_', date, '.csv'];
% writematrix(init_points, init_filename);


%% Read experiment data from file
exp_file = uigetfile('*.csv');
exp_table = readtable(exp_file, 'ReadVariableNames', false);
exp_data = table2array(exp_table);

%% Run BO/ChIDDO

% Run BO for each of the alternative models

X_known = exp_data(:,1:dims);
y_known = exp_data(:,dims+1);
% tot_batches = total # of batches before total_num is reached
tot_batches = floor((total_num - init_num)/batch_size);
curr_batch = floor((length(X_known) - init_num)/batch_size) + 1;

% init_tradeoff = value that corresponds to exploration vs. exploitation.
% Decreases in value as batches increase
if strcmp(acq_fun,'MRB')
    tradeoff = linspace(2,0,tot_batches);
elseif strcmp(acq_fun,'UCB')
    tradeoff = linspace(4,0,tot_batches);
else
    tradeoff = logspace(-1.3,-7,tot_batches);
end

phys_params = params;

for batch = curr_batch:tot_batches
    
    if strcmp(BO_ChIDDO, 'ChIDDO')
        all_data = cat(2,X_known,y_known);
        all_data_tbl = array2table(all_data);
        all_data_tbl.Properties.VariableNames = {'On' 'Off' 'FE'};
        options = statset('MaxIter', 500, 'TolFun', 1e-10, 'TolX', 1e-10);
        mdl = fitnlm(all_data_tbl, @(b,x)electro_2_reg(b,x), phys_params, 'Options',options);
        params_table = mdl.Coefficients.Estimate;
        phys_params = params_table.';
    end
    
    if strcmp(BO_ChIDDO,'ChIDDO')
        [X_used, y_used] = get_phys_points(X_known, y_known, total_num, phys_params, LB, UB);
        y_used(y_used < 0) = 0.01;
        y_used(y_used > 1) = 0.99;
    end

    new_tradeoff = tradeoff(batch);

    if strcmp(BO_ChIDDO, 'BO')
        % X_tot, X_known, y_known, batch_size, tradeoff, LB, UB, acq_name
        [X_new, preds, stds, MAE] = run_learner(X_tot, X_known, y_known, batch_size, new_tradeoff, LB, UB, acq_fun);
    else
        [X_new, preds, stds] = run_learner(X_tot, X_used, y_used, batch_size, new_tradeoff, LB, UB, acq_fun);
    end
    
    X_new;
    
    MAE
    
    X_known = cat(1,X_known,X_new);
    y_new = zeros(batch_size,1);
    y_known = cat(1,y_known,y_new);

    known_data = cat(2,X_known,y_known);

%     known_filename = ['experiment_data_batch_' num2str(batch) '_' BO_ChIDDO '_' date '.csv'];
%     writematrix(known_data, known_filename);
% 
    preds_filename = ['preds_batch_' num2str(batch) '_' BO_ChIDDO '_' date '.csv'];
    writematrix(preds, preds_filename);
% 
%     stds_filename = ['std_batch_' num2str(batch) '_' BO_ChIDDO '_' date '.csv'];
%     writematrix(stds, stds_filename);
%     
%     phys_params_filename = ['phys_params_' num2str(batch) '_' BO_ChIDDO '_' date '.csv'];
%     writematrix(phys_params, phys_params_filename);
    
    exit_val = 0;
    while exit_val == 0
        cont = input('Do you want to continue to the next batch? (Y/N) ', 's');
        if cont == 'Y'
            exit_val = 1;
            continue
        elseif cont == 'N'
            break
        else
            cont = input('Invalid input! Y/N');
        end
    end

    %%
%     known_data(end-batch_size+1:end,1) = [30;5;5];
%     known_data(end-batch_size+1:end,2) = [118;118;102];
%     figure();
%     scatter(X_tot(:,1), X_tot(:,2), 100, preds, 'filled');
%     colorbar();
%     hold on
% %     scatter(known_data(1:end-batch_size,1), known_data(1:end-batch_size,2),150,known_data(1:end-batch_size,3),'filled', 'MarkerEdgeColor', 'black', 'LineWidth',3);
% %     scatter(known_data(end-batch_size+1:end,1), known_data(end-batch_size+1:end,2),150,'red','filled');
%     xlabel('Active pulse time [ms]', 'FontSize',35);
%     ylabel('Resting pulse time [ms]', 'FontSize',35);
%     pbaspect([1 1 1]);
    
%     y_max_vals = calc_y_max(known_data(1:end-batch_size,3));
%     figure();
%     plotMMS(1:length(y_max_vals),y_max_vals, '-', 'black');
%     hold on
%     plotMMS(1:length(y_max_vals), known_data(1:end-batch_size,3), '*', 'blue');
% %     plot(1:length(y_max_vals),y_max_vals, 'LineWidth', 5);
%     xlabel('Experiment number');
%     ylabel('Faradaic Efficiency');
    
    if cont == 'N'
        break
    end
       
end

%%
figure();
scatter(X_tot(:,1)-2.5, X_tot(:,2)-2.5, 500, stds, 'filled');
h = colorbar();
ylabel(h, 'Standard Deviation');
% ylabel(h, 'Faradaic Efficiency');
hold on
% scatter(known_data(1:end-batch_size,1), known_data(1:end-batch_size,2),300,known_data(1:end-batch_size,3),'filled', 'MarkerEdgeColor', 'black', 'LineWidth',3);
scatter(known_data(1:end-batch_size,1), known_data(1:end-batch_size,2),300,'red','filled');
xlabel('Active pulse time [ms]');
ylabel('Resting pulse time [ms]');
set(gca,'FontName','Arial','FontSize',35,'linewidth',5,'TickLength',[0.025 0.025]);
box on
pbaspect([1 1 1]);

y_max_vals = calc_y_max(known_data(11:end-batch_size,3));
figure();
plotMMS(11:length(y_max_vals)+10,y_max_vals, '-', 'black');
hold on
plotMMS(1:length(known_data)-3, known_data(1:end-batch_size,3), '*', 'blue');
plotMMS(1:length(known_data)-3, ones(length(known_data)-3,1)*0.546857855, '-', 'red');
%     plot(1:length(y_max_vals),y_max_vals, 'LineWidth', 5);
xlabel('Experiment number');
ylabel('Faradaic Efficiency');
ylim([0 1]);
xlim([0 26]);


%%
% date = '20220125';
% date_preds = '2022-02-24';
% 
% BO_ChIDDO = 'BO';
% 
% mov(6) = struct('cdata',[],'colormap',[]);
% for j = 1:6
%     preds_filename = ['preds_batch_' num2str(j) '_' BO_ChIDDO '_' date_preds '.csv'];
%     preds_table = readtable(preds_filename, 'ReadVariableNames', false);
%     preds_data = table2array(preds_table);
%     
%     known_filename = [date '_batch_' num2str(j-1) '_data_' BO_ChIDDO '.csv'];
%     known_table = readtable(known_filename, 'ReadVariableNames', false);
%     known_data = table2array(known_table);
% 
%     figure();
%     scatter(X_tot(:,1)-2.5, X_tot(:,2)-2.5, 500, preds_data, 'filled');
%     h = colorbar();
%     ylabel(h, 'Faradaic Efficiency');
%     hold on
%     caxis([0.4 0.95]);
%     scatter(known_data(:,1), known_data(:,2),300,known_data(:,3),'filled', 'MarkerEdgeColor', 'black', 'LineWidth',3);
%     %     scatter(known_data(end-batch_size+1:end,1), known_data(end-batch_size+1:end,2),150,'red','filled');
%     xlabel('Active pulse time [ms]');
%     ylabel('Resting pulse time [ms]');
%     set(gca,'FontName','Arial','FontSize',35,'linewidth',5,'TickLength',[0.025 0.025]);
%     box on
%     pbaspect([1 1 1]);
%     
%     set(gcf, 'Position', get(0, 'Screensize'));
%     drawnow
%     mov(j) = getframe(gcf);
%     
% end
% 
% figure();
% set(gcf, 'Position', get(0, 'Screensize'));
% movie(gcf, mov, 2, 1);
% mov_2 = [mov, mov];
% 
% v = VideoWriter('Predictions_2D.avi');
% v.FrameRate = 1;
% open(v)
% writeVideo(v,mov_2)
% close(v)
