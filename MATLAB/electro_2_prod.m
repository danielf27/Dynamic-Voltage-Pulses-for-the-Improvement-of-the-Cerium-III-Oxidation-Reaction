function prod_4 = electro_2_prod(params, x)

Nx = 20;
n = Nx;

d = 100e-6;
Volt = x(3);
Init_A = 300;

E0_1 = 1.6;
E0_2 = 1.23;

j0_1 = params(1);
j0_2 = params(2);

alpha_1 = 0.5;
alpha_2 = 0.5;

D = params(3);
z = 3;

% params = readmatrix('Params_2020-06-07_10_alts_cerium.csv');

F = 96485;
R=8.314;
T=300;

Init = zeros(3*Nx,1);
Init(1:Nx) = Init_A*ones(Nx, 1);
% times = np.linspace(0,15)
% j0_1, j0_2, j0_3, j0_4, alpha_1,alpha_2,alpha_3, alpha_4, E0_1, E0_2,E0_3, E0_4, init_A, init_B, d, volt
opts = odeset('InitialStep', 1e-5, 'RelTol', 0.1 ,'AbsTol', 1e-2);
[time, conc] = ode15s(@(t,y) reaction_test_prod(t, y, x(1)/1000, x(2)/1000, j0_1, j0_2, alpha_1,alpha_2, E0_1, E0_2, D, d, Volt),[0, 5], Init, opts);
prod_4 = conc(end,2*Nx)/5;


end