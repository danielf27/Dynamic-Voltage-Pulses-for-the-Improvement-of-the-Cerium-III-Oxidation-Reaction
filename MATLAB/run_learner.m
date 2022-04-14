function [X_new, preds, stds, MAE] = run_learner(X_tot, X_known, y_known, batch_size, tradeoff, LB, UB, acq_name)
data_array = cat(2,X_known,y_known);
data_table = array2table(data_array);
if size(X_known,2) == 2
    data_table.Properties.VariableNames = {'x_1','x_2','y'};
elseif size(X_known,2) == 3
    data_table.Properties.VariableNames = {'x_1','x_2','x_3','y'};
else
    data_table.Properties.VariableNames = {'x_1','x_2','x_3','x_4','y'};
end

GPR_model = fitrgp(data_table,'y','KernelFunction','squaredexponential',...
  'FitMethod','exact','PredictMethod','exact','Standardize',1);

exp_table = readtable('20220125_batch_5_data_BO.csv', 'ReadVariableNames', false);
exp_data = table2array(exp_table);
X_exp = exp_data(:,1:2);

[preds, stds] = predict(GPR_model, X_tot);
[preds_exp, stds_exp] = predict(GPR_model, X_exp);
AE = 0;
for j=1:length(preds_exp)
    AE = AE + abs(preds_exp(j)-exp_data(j,3));
end
MAE = AE/(length(preds_exp));

max_pred = max(preds);
X_new = acq_select(X_tot, preds, stds, tradeoff, max_pred, batch_size, X_known, y_known, LB, UB, GPR_model, acq_name);

end