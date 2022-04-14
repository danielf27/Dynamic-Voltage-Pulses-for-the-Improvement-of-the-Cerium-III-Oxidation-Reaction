function hyp = hypervol_2D(p_points, ref_norm)

hyp=0;
for k = 1:length(p_points)
    side_1 = ref_norm(1) - p_points(k,2);
    if k == length(p_points)
        side_2 = ref_norm(2) - p_points(k,1);
    else
        side_2 = p_points(k+1,1) - p_points(k,1);
    end
    hyp = hyp + side_1*side_2;
end


end