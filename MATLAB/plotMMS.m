function plotMMS(x,y,linetype, color)
%Created by Miguel Modestino, 08/02/2017    
%Plots thick borders and large letters, standard for papers
%Inputs:
%x = values of x
%y = values of y
%linetype = '-', '--','-.-','.',+' etc

g=plot(x,y,linetype);
set(g,'linewidth',15,'markersize',15);
set(gca,'FontName','Arial','FontSize',35,'linewidth',5,'TickLength',[0.025 0.025]);
set(g, 'Color', color);
box on
pbaspect([1 1 1]);
end