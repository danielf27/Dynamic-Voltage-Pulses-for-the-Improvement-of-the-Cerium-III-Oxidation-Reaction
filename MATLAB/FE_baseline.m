X = [2 2.5 3 3.5 4];
FE = [0.902017982, 0.546857855, 0.369987707, 0.322342745, 0.245860712];
Curr = [38.99714026, 43.35296622, 55.03594132, 70.84983459, 104.9514115];
FE_std = [0.057770257, 0.034504917, 0.037035917, 0.006858613, 0.026168557];
Curr_std = [6.395682614, 5.500490496, 2.74060681, 1.145779562, 9.473426595];

figure();
yyaxis left
errorbar(X,FE,FE_std, '-k', 'LineWidth', 5);
hold on
plotMMS(X,FE,'*-','black');
xlabel('Voltage [V]');
ylabel('Faradaic Efficiency');
xlim([1.75,4.25]);
ylim([0,1]);
ax = gca;
ax.YColor = 'k';

yyaxis right
errorbar(X,Curr,Curr_std, '-b', 'LineWidth', 5);
plotMMS(X,Curr,'*-','blue');
ylabel('Partial Current Density [mA cm^{-2}]');
xlim([1.75,4.25]);
ylim([0,120]);
ax = gca;
ax.YColor = 'b';
