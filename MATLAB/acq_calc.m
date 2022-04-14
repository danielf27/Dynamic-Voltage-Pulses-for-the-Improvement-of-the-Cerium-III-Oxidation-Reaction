function acq_score = acq_calc(X0, mu, sigma, tradeoff, max_val, fun_name, model, X_known, y_known, X_tot)

[test_pred, test_std] = predict(model, X0);
switch fun_name
    case 'EI'
        z_test = -(test_pred - max_val - tradeoff)./test_std;
        z_all = -(mu - max_val - tradeoff)./sigma;
        ave_z = z_test;
        std_z = std(z_all);
        pd = makedist('Normal', 'mu',ave_z, 'sigma',std_z);
        acq_score = -(test_pred - max_val - tradeoff).*normcdf(z_test, ave_z, std_z) + test_std.*pdf(pd,z_test);
    case 'PI'
        z = -(test_pred - max_val - tradeoff)./test_std;
        acq_score = -normcdf(z, test_pred, test_std);
    case 'UCB'
        acq_score = -(test_pred + tradeoff.*test_std);
    case 'MRB'
        b = tradeoff;
        
        [grid_pred, grid_std] = predict(model, X_tot);
        
        grid_pred_max = max(grid_pred);
        grid_pred_min = min(grid_pred);
        pred_range = grid_pred_max - grid_pred_min;
        
        test_pred_norm = (test_pred - grid_pred_min)./pred_range;
        
        X_grid_max = max(X_tot);
        X_grid_min = min(X_tot);
        X_grid_range = X_grid_max - X_grid_min;
        
        X0_norm = (X0 - X_grid_min)./X_grid_range;
        X_known_norm = (X_known - X_grid_min)./X_grid_range;
        
        combined_X = cat(1,X0_norm, X_known_norm);
        p_dist = pdist(combined_X);
        square = squareform(p_dist);
        [min_val, ind] = min(square(2:end,1));
        
        DIST = min_val;
        STD = test_std;
        similarity = 1/(1+DIST);
        
        if DIST < 0
            DIST = 0;
        end
        acq_score = -(b*(1-similarity) + b*STD + test_pred_norm);
        
        
end

end