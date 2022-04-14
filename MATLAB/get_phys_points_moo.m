function [X_all, y_all] = get_phys_points_moo(X_known, y_known, num, params, LB, UB)

dims = size(X_known,2);
num_added = num - length(X_known);
choices = zeros(num_added, dims);
for j = 1:dims
    choices(:,j) = (UB(j)-LB(j)).*rand(num_added,1) + LB(j);
end

y_choices_1 = electro_2_reg(params, choices);
y_choices_2 = electro_2_reg_prod(params, choices);

min_choices_1 = min(y_choices_1);
max_choices_1 = max(y_choices_1);
min_choices_2 = min(y_choices_2);
max_choices_2 = max(y_choices_2);

% Normalize y_choices to y_known values
max_y_1 = max(y_known(:,1));
min_y_1 = min(y_known(:,1));
max_y_2 = max(y_known(:,2));
min_y_2 = min(y_known(:,2));
y_choices_1_norm = min_y_1 + (max_y_1-min_y_1)*((y_choices_1-min_choices_1)/(max_choices_1-min_choices_1));
y_choices_2_norm = min_y_2 + (max_y_2-min_y_2)*((y_choices_2-min_choices_2)/(max_choices_2-min_choices_2));

y_choices = [y_choices_1_norm y_choices_2_norm];


X_all = cat(1,X_known, choices);
y_all = cat(1,y_known, y_choices);

end