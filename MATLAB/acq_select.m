function X_new = acq_select(X_tot, mu, sigma, tradeoff, max_val, batch_size, X_known, y_known, LB, UB, model, acq_name)
GPR_model = model;
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
        [best_X(j,:), val(j)] = fmincon(@(x) acq_calc(x, mu, sigma, tradeoff, max_val, acq_name, GPR_model, X_known, y_known, X_tot), X0, [], [], [], [], LB, UB, [], options);
    end
    [max_val, max_ind] = min(val);
    X_new_temp = best_X(max_ind,:);
    X_new(batch,:) = X_new_temp;
    X_known = [X_known; X_new_temp];
    y_new = predict(model,best_X(max_ind,:));
    y_known = [y_known; y_new];
    
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

    [mu, sigma] = predict(GPR_model, X_tot);
    max_val = max(mu);
    

end