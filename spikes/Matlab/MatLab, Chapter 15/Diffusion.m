%Diffusion in one Dimension
clear all;
whitebg('w');
X = -1:0.01:1;
CC = zeros(1, length(X));
figure(1), WW = plot(X, CC, 'r-'); set(WW, 'LineWidth', 3); axis([-1 1 0 6]);
Film = moviein(30);

for TT = 1:30;
    Tm = TT * 0.1 - 0.1;
    CC = exp(-0.2 * Tm) * (3 + cos(2 * pi * X) * exp(-0.1 * Tm) + 2 * cos(2 * 5 * pi * X) * exp(-2.5 * Tm));
    WW = plot(X, CC, 'r-'); set(WW, 'LineWidth', 3); axis([-1 1 0 6]);
    Film(:, TT) = getframe;
end;

movie(Film, 2);
