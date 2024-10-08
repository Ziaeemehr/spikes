%Transparent & Coherent motion demo
Slope = input('Directions (1) �22� or (2) �68�: ');
if Slope == 1; SX = 1; SY = 2.5; else SX = 2.5; SY = 1; end;
[X, Y] = meshgrid(-100:100);
Apertures = ((X + 25) .^ 2 + (Y + 25) .^ 2) < 20 ^ 2;
Apertures = Apertures + (((X - 25) .^ 2 + (Y + 25) .^ 2) < 20 ^ 2);
Apertures = Apertures + (((X + 25) .^ 2 + (Y - 25) .^ 2) < 20 ^ 2);
Apertures = Apertures + (((X - 25) .^ 2 + (Y - 25) .^ 2) < 20 ^ 2);
Lines = 100 * cos(2 * pi * (X + Y) / 25);
figure(1), image(Lines); axis square; axis off; colormap(gray(200));
Motion = moviein(40);

for T = 1:40; %amination
    Lines = 100 * cos(2 * pi * (-SX * X .* sign(X) .* sign(Y) + SY * Y + 2 * T) / 40);
    Lines = Lines .* (Apertures) + 100;
    Lines(80:120, 10:50) = Lines(55:95, 55:95);
    Lines(10:50, 80:120) = Lines(55:95, 105:145);
    Lines(80:120, 150:190) = Lines(55:95, 105:145);
    Lines(150:190, 80:120) = Lines(55:95, 55:95);
    Lines(20:60, 20:60) = Lines(55:95, 105:145);
    Lines(140:180, 140:180) = Lines(55:95, 105:145);
    Lines(20:60, 140:180) = Lines(55:95, 55:95);
    Lines(140:180, 20:60) = Lines(55:95, 55:95);
    image(Lines); axis square; axis off; colormap(gray(200));
    Motion(:, T) = getframe;
end;

movie(Motion, 5);
