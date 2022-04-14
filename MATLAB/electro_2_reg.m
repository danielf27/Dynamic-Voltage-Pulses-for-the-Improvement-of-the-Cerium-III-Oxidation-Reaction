function vals = electro_2_reg(params, x)

vals = zeros(length(x),1);
for j = 1:length(x)
    vals(j) = electro_2(params, x(j,:));
    if imag(vals(j)) ~= 0
        vals(j) = 0.01;
    end
end
vals(vals < 0) = 0.01;
vals(vals > 1) = 0.99;
