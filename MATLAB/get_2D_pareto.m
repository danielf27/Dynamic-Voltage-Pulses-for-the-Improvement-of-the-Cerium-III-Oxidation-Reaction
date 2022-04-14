function [p_points, p_inds] = get_2D_pareto(y_vals)

[y_sort, index] = sort(y_vals);
y_sorted_1 = y_vals(index(:,1), :);

p_points = [];
p_inds = [];
p_ind = 0;
for j = 1:length(y_sorted_1)
    if j == 1
        p_points = [p_points, y_sorted_1(j,:)];
        p_inds = [p_inds; index(j, 1)];
        p_points = reshape(p_points,[],2);
        p_ind = p_ind + 1;
        continue
    end
    
    if y_sorted_1(j,2) < p_points(p_ind,2)
        p_points = [p_points; y_sorted_1(j,:)];
        p_inds = [p_inds; index(j,1)];
        p_points = reshape(p_points,[],2);
        p_ind = p_ind + 1;
    end
    
end



end