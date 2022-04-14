function acq_score = acq_calc_moo(X0, tradeoff, fun_name, model, model_2, X_known, y_known, X_tot)
% x, preds_1, std_1, preds_2, std_2, tradeoff, acq_name, GPR_model_1, GPR_model_2, X_known, y_known, X_tot
[test_pred, test_std] = predict(model, X0);
[test_pred_2, test_std_2] = predict(model_2, X0);
test_pred_tot = [test_pred, test_pred_2];
test_std_tot = [test_std, test_std_2];

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
        [grid_pred_2, grid_std_2] = predict(model_2, X_tot);
        grid_pred_tot = [grid_pred, grid_pred_2];
        
        grid_pred_max = max(grid_pred_tot);
        grid_pred_min = min(grid_pred_tot);
        pred_range = grid_pred_max - grid_pred_min;
        
        grid_pred_norm = (grid_pred_tot - grid_pred_min)./pred_range;
        grid_pred_norm = 1-grid_pred_norm;
        test_pred_norm = (test_pred_tot - grid_pred_min)./pred_range;
        test_pred_norm = 1-test_pred_norm;
        y_known_norm = (y_known - grid_pred_min)./pred_range;
        y_known_norm = 1-y_known_norm;
        
        combined_y = cat(1,test_pred_norm, y_known_norm);
        p_dist_y = pdist(combined_y);
        square_y = squareform(p_dist_y);
        [min_obj, ind] = min(square_y(2:end,1));
        
        X_grid_max = max(X_tot);
        X_grid_min = min(X_tot);
        X_grid_range = X_grid_max - X_grid_min;
        
        X0_norm = (X0 - X_grid_min)./X_grid_range;
        X_known_norm = (X_known - X_grid_min)./X_grid_range;
        
        combined_X = cat(1,X0_norm, X_known_norm);
        p_dist = pdist(combined_X);
        square = squareform(p_dist);
        [min_val, ind] = min(square(2:end,1));
        
        ref_point = [1 1];
        p_points_bef = get_2D_pareto(y_known_norm);
        hyp_before = hypervol_2D(p_points_bef, ref_point);
        
        temp = [y_known_norm; test_pred_norm];
        p_points_aft = get_2D_pareto(temp);
        hyp_after = hypervol_2D(p_points_aft, ref_point);
        
        p_points_grid = get_2D_pareto(grid_pred_norm);
        hyp_grid = hypervol_2D(p_points_grid, ref_point);
        
        max_hyp_dif = hyp_grid - hyp_before;
        
        DIST = min_val;
        hyp_dif = (hyp_after-hyp_before)/max_hyp_dif;
        STD = mean(test_std_tot);
        similarity = 1/(1+DIST);
        DIST_obj = min_obj;
        if DIST_obj < 0
            DIST_obj = 0;
        end
        d_obj_score = 1/(1+DIST_obj);
        
        if DIST < 0
            DIST = 0;
        end
        acq_score = -(1*b*(1-similarity) + 1*b*(1-d_obj_score) + b*STD + hyp_dif);
%         acq_score = -(hyp_dif);
end

end