function X_new = acq_select_moo(X_tot, preds_1, std_1, preds_2, std_2, tradeoff, batch_size, X_known, y_known, LB, UB, model_1, model_2, acq_name)
%                           X_tot, preds, stds, tradeoff, max_pred, batch_size, X_known, y_known, LB, UB, GPR_model, GPR_model_2, acq_name
GPR_model_1 = model_1;
GPR_model_2 = model_2;
dims = size(X_known,2);
X_new = zeros(batch_size,dims);
best_X = zeros(50,dims);
val = zeros(50,1);
for batch = 1:batch_size
    
    % Initialize the minimization 50 times to get closer to a global
    % minimum
    for j = 1:50
        X0 = zeros(1,dims);
        for col = 1:dims
            X0(col) = (UB(col)-LB(col)).*rand + LB(col);
        end
        options = optimset('TolFun', 1e-8, 'TolX', 1e-8);
        [best_X(j,:), val(j)] = fmincon(@(x) acq_calc_moo(x, tradeoff, acq_name, GPR_model_1, GPR_model_2, X_known, y_known, X_tot), X0, [], [], [], [], LB, UB, [], options);
    end
    [max_val, max_ind] = min(val);
    X_new_temp = best_X(max_ind,:);
    X_new(batch,:) = X_new_temp;
    X_known = [X_known; X_new_temp];
    y_new = zeros(1, 2);
    y_new(1) = predict(model_1,best_X(max_ind,:));
    y_new(2) = predict(model_2,best_X(max_ind,:));
    y_known = [y_known; y_new];
    
    data_array_1 = cat(2,X_known,y_known(:,1));
    data_table_1 = array2table(data_array_1);
    data_array_2 = cat(2,X_known,y_known(:,2));
    data_table_2 = array2table(data_array_2);
    if size(X_known,2) == 2
        data_table.Properties.VariableNames = {'x_1','x_2','y'};
    elseif size(X_known,2) == 3
        data_table_1.Properties.VariableNames = {'x_1','x_2','x_3','y'};
        data_table_2.Properties.VariableNames = {'x_1','x_2','x_3','y'};
    else
        data_table.Properties.VariableNames = {'x_1','x_2','x_3','x_4','y'};
    end

    GPR_model_1 = fitrgp(data_table_1,'y','KernelFunction','squaredexponential',...
      'FitMethod','exact','PredictMethod','exact','Standardize',1);
  
    GPR_model_2 = fitrgp(data_table_2,'y','KernelFunction','squaredexponential',...
      'FitMethod','exact','PredictMethod','exact','Standardize',1);

    [preds_1, std_1] = predict(GPR_model_1, X_tot);
    [preds_2, std_2] = predict(GPR_model_2, X_tot);
    

end