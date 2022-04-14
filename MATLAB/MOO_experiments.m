%% Multi-Objective Bayesian Optimization for experiments (MATLAB)

%% Define hyperparameters
% If any of the dimensions have a different range, input LB and UB as
% below:
% LB = [0.1 0.1 0.1];
% UB = [3 2 2];
% If all the dimensions have the same range, use single values as below:
LB = [5, 5, 2];
UB = [200, 200, 4];

date = '2022-03-08';

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
num = 15;

% Choose from MRB, PI, UCB, EI
acq_fun = 'MRB';

% Choose from BO, ChIDDO
BO_ChIDDO = 'BO';

 
%% Create comparison grid of design space
x_1 = linspace(LB(1),UB(1),num);
x_2 = linspace(LB(2),UB(2),num);
x_3 = linspace(LB(3), UB(3), num);
X_tot = zeros(num^dims,dims);
X_tot_num = zeros(num^dims,dims+1);

for j = 1:length(x_1)
    for k = 1:length(x_2)
        for m = 1:length(x_3)
    % If adding another dimension, n would be:
            n = num*num*(j-1) + num*(k-1) + m;
%         n = num*(j-1) + k;
            X_tot(n, :) = [x_1(j) x_2(k) x_3(m)];
            X_tot_num(n, :) = [x_1(j) x_2(k) x_3(m) n];
        end
    end
end

%% Define parameters/parameter ranges and create a set of alternative parameters
% params = the physics model parameters that is your estimated parameters
% of the system
params = [1e-8, 1e-9, 5e-11];
% param_std = estimated standard deviation of the possible parameter
% values. This is used to create alternate models for testing purposes.
% param_std = [0.05, 0.05, 2e-8, 2e-10, 0.15, 0.15];

% data = zeros(length(X_tot), 2);
% for j = 1:length(data)
%     data(j,1) = electro_2(params, X_tot(j,:));
%     data(j,2) = electro_2_prod(params, X_tot(j,:));
% end

%%
% figure;
% scatter3(X_tot(:,1), X_tot(:,2), X_tot(:,3), 200, data(:,1));
% colorbar();
% 
% figure;
% scatter3(X_tot(:,1), X_tot(:,2), X_tot(:,3), 200, data(:,2));
% colorbar();
% Generate alternate parameter sets for testing purposes.

% Select initial experiments
% if length(UB)==1
%     init_points = (UB-LB).*rand(init_num,dims) + LB;
%     % If you're variable only goes to a certain decimal point, use the
%     % round function below:
% %     init_points = round(init_points, 1);
% else
%     init_points = zeros(init_num, dims);
%     for j = 1:length(UB)
%         init_points(:,j) = (UB(j)-LB(j)).*rand(init_num,1) + LB(j);
%     end
% end
% 
% % Save the initial points in a file
% init_filename = ['Init_points_3D_opt_', date, '.csv'];
% writematrix(init_points, init_filename);


%% Read experiment data from file
exp_file = uigetfile('*.csv');
exp_table = readtable(exp_file, 'ReadVariableNames', false);
exp_data = table2array(exp_table);

%% Run BO/ChIDDO

% Run BO for each of the alternative models

X_known = exp_data(:,1:dims);
y_known = exp_data(:,dims+1:end);
% tot_batches = total # of batches before total_num is reached
tot_batches = floor((total_num - init_num)/batch_size);
curr_batch = floor((length(X_known) - init_num)/batch_size) + 1;

% init_tradeoff = value that corresponds to exploration vs. exploitation.
% Decreases in value as batches increase
if strcmp(acq_fun,'MRB')
    tradeoff = linspace(1,0,tot_batches);
elseif strcmp(acq_fun,'UCB')
    tradeoff = linspace(4,0,tot_batches);
else
    tradeoff = logspace(-1.3,-7,tot_batches);
end

phys_params = params;

for batch = curr_batch:tot_batches
    
    if strcmp(BO_ChIDDO, 'ChIDDO')
        all_data_1 = cat(2,X_known,y_known(:,1));
        all_data_tbl_1 = array2table(all_data_1);
        all_data_tbl_1.Properties.VariableNames = {'On' 'Off' 'Volt' 'FE'};
        all_data_2 = cat(2,X_known,y_known(:,2));
        all_data_tbl_2 = array2table(all_data_2);
        all_data_tbl_2.Properties.VariableNames = {'On' 'Off' 'Volt' 'Prod'};
        options = statset('MaxIter', 500);
        mdl_1 = fitnlm(all_data_tbl_1, @(b,x)electro_2_reg(b,x), phys_params, 'Options',options);
        mdl_2 = fitnlm(all_data_tbl_2, @(b,x)electro_2_reg_prod(b,x), phys_params, 'Options',options);
        params_table_1 = mdl_1.Coefficients.Estimate;
        params_table_2 = mdl_2.Coefficients.Estimate;
        phys_params_1 = params_table_1.';
        phys_params_2 = params_table_2.';
%         phys_params = mean([phys_params_1; phys_params_2]);
        phys_params = phys_params_2;
    end
    
    if strcmp(BO_ChIDDO,'ChIDDO')
        [X_used, y_used] = get_phys_points_moo(X_known, y_known, total_num, phys_params, LB, UB);
%         y_used(y_used < 0) = 0.01;
%         y_used(y_used > 1) = 0.99;
    end

    new_tradeoff = tradeoff(batch);

    if strcmp(BO_ChIDDO, 'BO')
        % X_tot, X_known, y_known, batch_size, tradeoff, LB, UB, acq_name
        [X_new, preds, stds, y_pred] = run_learner_moo(X_tot, X_known, y_known, batch_size, new_tradeoff, LB, UB, acq_fun);
    else
        [X_new, preds, stds, y_pred] = run_learner_ChIDDO_moo(X_tot, X_known, y_known, X_used, y_used, batch_size, new_tradeoff, LB, UB, acq_fun);
    end
    
    X_new
    
    X_known = cat(1,X_known,X_new);
    y_new = zeros(batch_size,2);
    y_known = cat(1,y_known,y_new);

    known_data = cat(2,X_known,y_known);

    known_filename = ['experiment_data_batch_' num2str(batch) '_' BO_ChIDDO '_' date '.csv'];
    writematrix(known_data, known_filename);

    preds_filename = ['preds_batch_' num2str(batch) '_' BO_ChIDDO '_' date '.csv'];
    writematrix(preds, preds_filename);

    stds_filename = ['std_batch_' num2str(batch) '_' BO_ChIDDO '_' date '.csv'];
    writematrix(stds, stds_filename);
    
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
    
%     figure();
%     scatter(X_tot(:,1), X_tot(:,2), 100, preds,'filled');
%     colorbar();
%     hold on
%     scatter(known_data(1:-3,1), known_data(1:-3,2),150,'black','filled');
%     xlabel('Active pulse time [ms]');
%     ylabel('Resting pulse time [ms]');
    
%     y_max_vals = calc_y_max(known_data(1:-3,3));
%     known_data(1:-3,3)
%     figure();
%     plot(1:length(y_max_vals),y_max_vals, 'LineWidth', 3);
%     xlabel('Experiment number');
%     ylabel('Faradaic Efficiency');
    
    if cont == 'N'
        break
    end
       
end

%% Plot data and save error percent and distance files

% figure();
% scatter3(X_tot(:,1),X_tot(:,2),X_tot(:,3), 400, preds(:,1), 'filled');
% set(gca,'FontName','Arial','FontSize',30,'linewidth',2,'TickLength',[0.025 0.025]);
% xlabel('Active pulse time [ms]', 'Rotation', 20);
% ylabel('Resting pulse time [ms]', 'Rotation', -30);
% zlabel('Voltage [V]');
% pbaspect([1 1 1]);
% h = colorbar();
% ylabel(h, 'Faradaic Efficiency');

% figure();
% scatter3(X_used(17:end,1),X_used(17:end,2),X_used(17:end,3), 400, y_used(17:end,2), 'filled');
% set(gca,'FontName','Arial','FontSize',30,'linewidth',2,'TickLength',[0.025 0.025]);
% xlabel('Active pulse time [ms]', 'Rotation', 20);
% ylabel('Resting pulse time [ms]', 'Rotation', -30);
% zlabel('Voltage [V]');
% pbaspect([1 1 1]);
% h = colorbar();
% ylabel(h, 'Partial Current Density [mA cm^{-2}]');
% 
% figure();
% scatter3(X_used(17:end,1),X_used(17:end,2),X_used(17:end,3), 400, y_used(17:end,1), 'filled');
% set(gca,'FontName','Arial','FontSize',30,'linewidth',2,'TickLength',[0.025 0.025]);
% xlabel('Active pulse time [ms]', 'Rotation', 20);
% ylabel('Resting pulse time [ms]', 'Rotation', -30);
% zlabel('Voltage [V]');
% pbaspect([1 1 1]);
% h = colorbar();
% ylabel(h, 'Faradaic Efficiency');

%%

X_tot_low = X_tot_num(X_tot(:,3)==2,:);
X_tot_med = X_tot_num(X_tot(:,3)==3,:);
X_tot_hi = X_tot_num(X_tot(:,3)==4,:);
inds_low = X_tot_low(:,4);
inds_med = X_tot_med(:,4);
inds_hi = X_tot_hi(:,4);
preds_low = preds(inds_low, :);
preds_med = preds(inds_med, :);
preds_hi = preds(inds_hi, :);
[max_pred_hi, max_loc_hi] = max(preds_hi);

figure();
[X_1, X_2] = meshgrid(x_1, x_2);
contourf(X_1, X_2, reshape(preds_med(:,1),[15,15]), 20);
hold on

% scatter(X_tot_hi(max_loc_hi,1), X_tot_hi(max_loc_hi,2), 500, 'red', 'filled');
% imagesc(X_tot_hi(:,1), X_tot_hi(:,2), preds_hi(:,2));
% scatter(X_tot_hi(:,1),X_tot_hi(:,2), 3500, preds_hi(:,2), 'filled');
hold on
set(gca,'FontName','Arial','FontSize',30,'linewidth',2,'TickLength',[0.025 0.025]);
xlabel('Active pulse time [ms]');
ylabel('Resting pulse time [ms]');
pbaspect([1 1 1]);
h = colorbar();
% colormap(cool);
ylabel(h, 'Faradaic Efficiency');
caxis([min(preds(:,1)) max(preds(:,1))]);
% ylabel(h, 'Partial Current Density [mA cm^{-2}]');
xlim([5 200]);
ylim([5 200]);
% zlim([2 4]);


%%
figure();
scatter3(X_tot(:,1),X_tot(:,2),X_tot(:,3), 400, preds(:,2), 'filled');
set(gca,'FontName','Arial','FontSize',30,'linewidth',2,'TickLength',[0.025 0.025]);
xlabel('Active pulse time [ms]', 'Rotation', 20);
ylabel('Resting pulse time [ms]', 'Rotation', -30);
zlabel('Voltage [V]');
pbaspect([1 1 1]);
h = colorbar();
ylabel(h, 'Partial Current Density [mA cm^{-2}]');

figure();
scatter3(X_tot(:,1),X_tot(:,2),X_tot(:,3), 400, preds(:,1), 'filled');
set(gca,'FontName','Arial','FontSize',30,'linewidth',2,'TickLength',[0.025 0.025]);
xlabel('Active pulse time [ms]', 'Rotation', 20);
ylabel('Resting pulse time [ms]', 'Rotation', -30);
zlabel('Voltage [V]');
pbaspect([1 1 1]);
h = colorbar();
ylabel(h, 'Faradaic Efficiency');
% 
% figure();
% plotMMS(y_known(1:end-3,1), y_known(1:end-3,2), '*', 'black');
% hold on
% % plotMMS(y_pred(:,1), y_pred(:,2), '*', 'red');
% % set(gca,'FontName','Arial','FontSize',20,'linewidth',5,'TickLength',[0.025 0.025]);
% xlabel('Faradaic Efficiency');
% ylabel('Partial Current Density [mA cm^{-2}]');
% xticks([0 0.25 0.5 0.75 1])
% xlim([0 1]);
% ylim([0 120]);

figure();
plotMMS(preds(:,1), preds(:,2), '*', 'black');
hold on
% scatter(y_pred(:,1), y_pred(:,2), 100, 'red', 'filled');
% set(gca,'FontName','Arial','FontSize',20,'linewidth',5,'TickLength',[0.025 0.025]);
xlabel('Faradaic Efficiency');
ylabel('Partial Current Density [mA cm^{-2}]');
xticks([0 0.25 0.5 0.75 1])
xlim([0 1]);
ylim([0 120]);

neg_preds = zeros(length(preds),2);
neg_preds(:,1) = 1-preds(:,1);
neg_preds(:,2) = 120 - preds(:,2);
[p_points, p_inds] = get_2D_pareto(neg_preds);
p_exps = X_tot(p_inds,:);

pareto_filename = ['pareto_inds_' num2str(curr_batch) '_' BO_ChIDDO '.csv'];
writematrix(p_inds, pareto_filename);

figure();
plotMMS(preds(:,1), preds(:,2), '*', 'black');
% plotMMS(preds(p_inds,1), preds(p_inds,2), '-', 'red');
hold on
% area(preds(p_inds,1), preds(p_inds,2), 'FaceColor', [1 0.6 0.6], 'EdgeColor', [1 0.6 0.6]); 
% area([0 preds(p_inds(end),1)], [preds(p_inds(end),2) preds(p_inds(end),2)], 'FaceColor', [1 0.6 0.6], 'EdgeColor', [1 0.6 0.6]);
% plotMMS(y_known(1:end-3,1), y_known(1:end-3,2), '*', 'black');
plotMMS(preds(p_inds,1), preds(p_inds,2), '*', 'red');
% set(gca,'FontName','Arial','FontSize',20,'linewidth',5,'TickLength',[0.025 0.025]);
xlabel('Faradaic Efficiency');
ylabel('Partial Current Density [mA cm^{-2}]');
xticks([0 0.25 0.5 0.75 1])
xlim([0 1]);
ylim([0 120]);

figure();
scatter3(p_exps(:,1),p_exps(:,2),p_exps(:,3), 400, preds(p_inds,2), 'filled');
set(gca,'FontName','Arial','FontSize',30,'linewidth',2,'TickLength',[0.025 0.025]);
xlabel('Active pulse time [ms]', 'Rotation', 20);
ylabel('Resting pulse time [ms]', 'Rotation', -30);
zlabel('Voltage [V]');
pbaspect([1 1 1]);
h = colorbar();
ylabel(h, 'Partial Current Density [mA cm^{-2}]');
xlim([0 200]);
ylim([0 200]);
zlim([2 4]);

figure();
scatter3(p_exps(:,1),p_exps(:,2),p_exps(:,3), 400, preds(p_inds,1), 'filled');
set(gca,'FontName','Arial','FontSize',30,'linewidth',2,'TickLength',[0.025 0.025]);
xlabel('Active pulse time [ms]', 'Rotation', 20);
ylabel('Resting pulse time [ms]', 'Rotation', -30);
zlabel('Voltage [V]');
pbaspect([1 1 1]);
h = colorbar();
ylabel(h, 'Faradaic Efficiency');
xlim([0 200]);
ylim([0 200]);
zlim([2 4]);

%%
figure();
scatter3(X_known(1:end-3,1),X_known(1:end-3,2),X_known(1:end-3,3), 400, y_known(1:end-3,1), 'filled');
set(gca,'FontName','Arial','FontSize',30,'linewidth',2,'TickLength',[0.025 0.025]);
xlabel('Active pulse time [ms]', 'Rotation', 20);
ylabel('Resting pulse time [ms]', 'Rotation', -30);
zlabel('Voltage [V]');
pbaspect([1 1 1]);
h = colorbar();
ylabel(h, 'Faradaic Efficiency');
xlim([0 200]);
ylim([0 200]);
zlim([2 4]);

figure();
scatter3(X_known(1:end-3,1),X_known(1:end-3,2),X_known(1:end-3,3), 400, y_known(1:end-3,2), 'filled');
set(gca,'FontName','Arial','FontSize',30,'linewidth',2,'TickLength',[0.025 0.025]);
xlabel('Active pulse time [ms]', 'Rotation', 20);
ylabel('Resting pulse time [ms]', 'Rotation', -30);
zlabel('Voltage [V]');
pbaspect([1 1 1]);
h = colorbar();
colormap(cool);
ylabel(h, 'Partial Current Density [mA cm^{-2}]');
xlim([0 200]);
ylim([0 200]);
zlim([2 4]);


%%
% date_1 = '2022-02-16';
% figure();
% for j = 1:7
%     pareto_filename = ['pareto_inds_' num2str(j) '_' BO_ChIDDO '.csv'];
%     p_inds_table = readtable(pareto_filename, 'ReadVariableNames', false);
%     p_inds_data = table2array(p_inds_table);
%     
%     if j<7
%         preds_filename = ['preds_batch_' num2str(j) '_' BO_ChIDDO '_' date_1 '.csv'];
%         preds_table = readtable(preds_filename, 'ReadVariableNames', false);
%         preds_data = table2array(preds_table);
%     else
%         preds_filename = ['preds_batch_' num2str(j) '_' BO_ChIDDO '_' date '.csv'];
%         preds_table = readtable(preds_filename, 'ReadVariableNames', false);
%         preds_data = table2array(preds_table);
%     end
%     
%     batch = ones(length(p_inds_data), 1)*j;
%     p_exps = X_tot(p_inds_data,:);
%     
%     scatter3(batch,preds_data(p_inds_data,1),preds_data(p_inds_data,2), 0.01);
%     hold on
%     plot3(batch,preds_data(p_inds_data,1),preds_data(p_inds_data,2), 'LineWidth', 5);
% end
% set(gca,'FontName','Arial','FontSize',30,'linewidth',2,'TickLength',[0.025 0.025]);
% xlabel('Batch', 'Rotation', 30);
% ylabel('Faradaic Efficiency', 'Rotation', -10);
% zlabel('Partial Current Density [mA cm^{-2}]');
% pbaspect([1 1 1]);
% ylim([0 1]);
% zlim([0 105]);


%%
dates = ["2022-02-16" "2022-02-16" "2022-02-16" "2022-02-16" "2022-02-16" "2022-02-16" "2022-02-23" "2022-03-08"];

BO_ChIDDO = 'BO';

mov(8) = struct('cdata',[],'colormap',[]);
for j = 1:8
    pareto_filename = ['pareto_inds_' num2str(j) '_' BO_ChIDDO '.csv'];
    p_inds_table = readtable(pareto_filename, 'ReadVariableNames', false);
    p_inds = table2array(p_inds_table);
    
    preds_filename = ['preds_batch_' num2str(j) '_' BO_ChIDDO '_' convertStringsToChars(dates(j)) '.csv'];
    preds_table = readtable(preds_filename, 'ReadVariableNames', false);
    preds_data = table2array(preds_table);
    
    known_filename = ['experiment_data_DC_batch_' num2str(j) '_' BO_ChIDDO '_' convertStringsToChars(dates(j)) '.csv'];
    known_table = readtable(known_filename, 'ReadVariableNames', false);
    known_data = table2array(known_table);

    figure();
    % plotMMS(preds(:,1), preds(:,2), '*', 'black');
    plotMMS(preds_data(p_inds,1), preds_data(p_inds,2), '-', 'red');
    hold on
    area(preds_data(p_inds,1), preds_data(p_inds,2), 'FaceColor', [1 0.6 0.6], 'EdgeColor', [1 0.6 0.6]); 
    area([0 preds_data(p_inds(end),1)], [preds_data(p_inds(end),2) preds_data(p_inds(end),2)], 'FaceColor', [1 0.6 0.6], 'EdgeColor', [1 0.6 0.6]);
    plotMMS(known_data(1:5,4), known_data(1:5,5), '*', 'magenta');
    plotMMS(known_data(6:end-3,4), known_data(6:end-3,5), '*', 'black');
    % scatter(y_pred(:,1), y_pred(:,2), 100, 'red', 'filled');
    % set(gca,'FontName','Arial','FontSize',20,'linewidth',5,'TickLength',[0.025 0.025]);
    xlabel('Faradaic Efficiency');
    ylabel('Partial Current Density [mA cm^{-2}]');
    xticks([0 0.25 0.5 0.75 1])
    xlim([0 1]);
    ylim([0 120]);
    set(gcf, 'Position', get(0, 'Screensize'));
    drawnow
    mov(j) = getframe(gcf);
    
end

% figure();
% set(gcf, 'Position', get(0, 'Screensize'));
% movie(gcf, mov, 2, 1);
% mov_2 = [mov, mov];
% 
% v = VideoWriter('Pareto_3D.avi');
% v.FrameRate = 1;
% open(v)
% writeVideo(v,mov_2)
% close(v)


%%

% Voltage = [2 2.5 3 3.5 4];
% FE_avg = [0.948777083, 0.546857855, 0.369987707, 0.322342745, 0.245860712];
% Curr_avg = [41.44741368, 43.35296622, 55.03594132, 70.84983459, 104.9514115];
% FE_std = [0.007565642, 0.034504917, 0.037035917, 0.006858613, 0.026168557];
% Curr_std = [11.0086532, 5.500490496, 2.74060681, 1.145779562, 9.473426595];
% 
% figure();
% yyaxis left
% errorbar(Voltage, FE_avg, FE_std, 'LineWidth', 8, 'Color', 'black');
% hold on
% plotMMS(Voltage, FE_avg, '-*', 'black');
% % plotMMS(Voltage, FE_avg, '*', 'black');
% ylabel('Faradaic Efficiency');
% xlabel('Voltage [V]');
% ax = gca;
% ax.YColor = 'k';
% xlim([1.75 4.25]);
% 
% yyaxis right
% errorbar(Voltage, Curr_avg, Curr_std, 'LineWidth', 8, 'Color', 'blue');
% plotMMS(Voltage, Curr_avg, '-*', 'blue');
% ax.YColor = 'b';
% ylabel('Partial Current Density [mA cm^{-2}]');







