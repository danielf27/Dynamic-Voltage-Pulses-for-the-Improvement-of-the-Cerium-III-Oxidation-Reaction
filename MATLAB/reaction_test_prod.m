function f = reaction_test_prod(t, c, t_on, t_off, j0_1, j0_2, alpha_1,alpha_2, E0_1, E0_2, D, d, volt)

Nx = 20;
dx = d/(Nx+1);
F = 96485; % C/mol
R = 8.314;
T = 300;
elec_dist = 0.01;
z=3;
Pot = volt/elec_dist;
dif_volt = ((volt-2)/2)*.1 + 2.45;
eta_1 = dif_volt-E0_1;
eta_2 = dif_volt-E0_2;
M = D*z*96485*Pot/(R*T);

% 6 species of interest, A, B, C, E, G, H
f = zeros(3*Nx, 1);

f(1) = 0;
f(Nx+1) = 0;
f(2*Nx+1) = 0;

Dec = t/(t_on + t_off);

if Dec - floor(Dec) < (t_on/(t_on + t_off))
    rrate_1 = j0_1*((c(Nx))^1)*exp((alpha_1*F*eta_1/R/T));
    rrate_2 = j0_2*exp((alpha_2*F*eta_2/R/T));
else
    rrate_1 = 0;
    rrate_2 = 0;
end

f(Nx) = ((-rrate_1)/(F*dx)) + (D/(dx^2))*(c(Nx-1) - c(Nx)) + (M/(dx))*(c(Nx-1) - c(Nx));
f(2*Nx) = ((rrate_1)/(F*dx)) + (D/(dx^2))*(c(2*Nx-1) - c(2*Nx)) + (M/(dx))*(c(2*Nx-1) - c(2*Nx));
f(3*Nx) = ((rrate_2)/(F*dx)) + (D/(dx^2))*(c(3*Nx-1) - c(3*Nx)) + (M/(dx))*(c(3*Nx-1) - c(3*Nx));

% Inner points
for j = 2:Nx-1
    f(j) = (D/(dx^2))*(c(j-1) - 2*c(j) + c(j+1)) + (M/(2*dx))*(c(j+1) - c(j-1));
    f(Nx+j) = (D/(dx^2))*(c(Nx+j-1) - 2*c(Nx+j) + c(Nx+j+1)) + (M/(2*dx))*(c(Nx+j+1) - c(Nx+j-1));
    f(2*Nx+j) = (D/(dx^2))*(c(2*Nx+j-1) - 2*c(2*Nx+j) + c(2*Nx+j+1)) + (M/(2*dx))*(c(2*Nx+j+1) - c(2*Nx+j-1));
end

end