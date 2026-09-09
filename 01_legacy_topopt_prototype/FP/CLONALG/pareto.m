%% Pareto-orientierte Darstellung der Endergebnisse
% x-Achse: mittleres Drehmoment
% y-Achse: Drehmomentwelligkeit

clear; clc; close all;

%% Daten aus Tabelle 4.1
labels = {'R0','R1','M1','M2','M3','M4'};

modelNames = {
    'Referenzmodell'
    'Diskretisiertes Referenzmodell'
    'Reparaturbasiert, N=4, 100 Gen.'
    'Reparaturbasiert, N=10, 100 Gen.'
    'Strafbasiert, N=10, 100 Gen.'
    'Strafbasiert, N=10, 1000 Gen.'
};

T_avg = [3.4828, 3.2240, 3.2456, 3.2040, 3.2280, 3.2120];
T_rip = [0.1177, 0.1232, 0.0802, 0.0874, 0.1136, 0.0925];

%% Plot
figure('Color','w');
hold on; grid on; box on;

% Alle Endergebnisse als Punkte
scatter(T_avg, T_rip, 70, 'filled');

% Beschriftung der Punkte
for i = 1:numel(labels)
    text(T_avg(i) + 0.004, T_rip(i), labels{i}, ...
        'FontSize', 11, ...
        'FontWeight', 'bold', ...
        'VerticalAlignment', 'middle');
end

%% Vereinfachte Pareto-Verbindung
% In den vorliegenden Endergebnissen sind vor allem R0 und M1 pareto-günstig:
% R0: höchstes mittleres Drehmoment
% M1: kleinste Drehmomentwelligkeit
idxPareto = [1, 3];   % R0 und M1

plot(T_avg(idxPareto), T_rip(idxPareto), '--', ...
    'LineWidth', 1.5);

%% Achsenbeschriftung
xlabel('Mittleres Drehmoment / Nm', 'FontSize', 12);
ylabel('Drehmomentwelligkeit / Nm', 'FontSize', 12);

title('Pareto-orientierte Darstellung der Endergebnisse', ...
    'FontSize', 13, ...
    'FontWeight', 'normal');

%% Achsengrenzen etwas schöner setzen
xlim([3.18, 3.50]);
ylim([0.075, 0.130]);

%% Hinweis für günstigere Richtung
% Text im Diagramm
text(3.36, 0.083, {'günstiger:', 'höheres Drehmoment', 'kleinere Welligkeit'}, ...
    'FontSize', 10, ...
    'HorizontalAlignment', 'left');

% Einfacher Richtungspfeil nach rechts unten
annotation('arrow', [0.63 0.75], [0.35 0.25]);

%% Legende
legend({'Endergebnisse', 'Pareto-orientierte Verbindung'}, ...
    'Location', 'northeast');

%% Optional: Tabelle in der Konsole ausgeben
resultTable = table(labels', modelNames, T_avg', T_rip', ...
    'VariableNames', {'Label', 'Modell', 'Mittleres_Drehmoment_Nm', 'Drehmomentwelligkeit_Nm'});

disp(resultTable);

%% Optional: Bild speichern
% Die Auflösung 300 dpi ist für Word normalerweise ausreichend.
exportgraphics(gcf, 'pareto_enderegebnisse.png', 'Resolution', 300);
exportgraphics(gcf, 'pareto_enderegebnisse.pdf', 'ContentType', 'vector');