function vals = electro_2_reg_prod(params, x)

vals = zeros(length(x),1);
for j = 1:length(x)
    vals(j) = electro_2_prod(params, x(j,:));
    if imag(vals(j)) ~= 0
        vals(j) = 1;
    end
end

