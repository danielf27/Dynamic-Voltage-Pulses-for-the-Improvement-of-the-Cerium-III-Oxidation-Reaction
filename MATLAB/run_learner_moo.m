function [X_new, preds, stds, y_pred] = run_learner_moo(X_tot, X_known, y_known, batch_size, tradeoff, LB, UB, acq_name)
data_array = cat(2,X_known,y_known(:,1));
data_array_2 = cat(2,X_known,y_known(:,2));
data_table = array2table(data_array);
data_table_2 = array2table(data_array_2);
if size(X_known,2) == 2
    data_table.Properties.VariableNames = {'x_1','x_2','y'};
elseif size(X_known,2) == 3
    data_table.Properties.VariableNames = {'x_1','x_2','x_3','y'};
    data_table_2.Properties.VariableNames = {'x_1','x_2','x_3','y'};
else
    data_table.Properties.VariableNames = {'x_1','x_2','x_3','x_4','y'};
end

GPR_model = fitrgp(data_table,'y','KernelFunction','squaredexponential',...
  'FitMethod','sr','PredictMethod','sr','Standardize',1);

GPR_model_2 = fitrgp(data_table_2,'y','KernelFunction','squaredexponential',...
  'FitMethod','sr','PredictMethod','sr','Standardize',1);

[preds_1, stds_1] = predict(GPR_model, X_tot);
[preds_2, stds_2] = predict(GPR_model_2, X_tot);
preds = cat(2,preds_1,preds_2);
stds = cat(2,stds_1,stds_2);
X_new = acq_select_moo(X_tot, preds_1, stds_1, preds_2, stds_2, tradeoff, batch_size, X_known, y_known, LB, UB, GPR_model, GPR_model_2, acq_name);
% X_tot, preds_1, std_1, preds_2, std_2, tradeoff, batch_size, X_known, y_known, LB, UB, model, acq_name

[y_preds_1, y_stds_1] = predict(GPR_model, X_new);
[y_preds_2, y_stds_2] = predict(GPR_model_2, X_new);
y_pred = cat(2,y_preds_1,y_preds_2);
end