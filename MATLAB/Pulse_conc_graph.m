% Pulse graph calculation

num = 40;

x_1 = linspace(5,200,num);
x_2 = linspace(5,200,num);
X_tot = zeros(num*num,2);

for j = 1:length(x_1)
    for k = 1:length(x_2)
        n = num*(j-1) + k;
        X_tot(n, :) = [x_1(j) x_2(k)];
    end
end

x = [100,0; 200,100; 200, 30; 100, 200];

params = [1e-8, 1e-9, 5e-10];
color = [0,0,0; 0 0 1;0 1 0;0 1 1];
FE = zeros(length(X_tot),1);
figure();
hold on
for j = 1:length(x)
    [time, conc] = electro_2_conc(params, x(j,:));
    plotMMS(time, conc(:,20), '-', color(j,:));
end
xlabel('Time [s]');
ylabel('Surface concentration [mol/m^3]');
legend('No Pulse', '200 ms on, 100 ms off', '200 ms on, 30 ms off', '100 ms on, 200 ms off');
ylim([0 300]);
% xlim([0 5]);


%%
% figure();
% % scatter(X_tot(:,1)-2.5, X_tot(:,2)-2.5, 10, FE, 'filled');
% % hold on
% im = imagesc(X_tot(:,1)-2.5, X_tot(:,2)-2.5, reshape(FE, 40,40));
% % imagesc(reshape(FE, 40,40));
% im.Interpolation = 'bilinear';
% h = colorbar();
% ylabel(h, 'Faradaic Efficiency');
% hold on
% xlabel('Active pulse time [ms]');
% ylabel('Resting pulse time [ms]');
% set(gca, 'YDir','normal');
% set(gca,'FontName','Arial','FontSize',35,'linewidth',5,'TickLength',[0.025 0.025]);
% box on
% pbaspect([1 1 1]);
    
    