%Generate Red & Green Binocular Rivalry Stimulus
%To be viewed with R&G glasses
XX = -150:150;
[X, Y] = meshgrid(XX);
Horiz = cos(2 * pi * (X + Y) / 32);
Vert = cos(2 * pi * (X - Y) / 32);
Horiz = 2 * (Horiz > 0);
Vert = (Vert > 0);
Pattern = Horiz + Vert + 1;
Radius = sqrt(X .^ 2 + Y .^ 2);
Pattern = Pattern .* (Radius < 127);
Cmap = [0 0 0; 0 1 0; 1 0 0; 1 1 0]; %Specify colors Red, Green & Yellow
figure (1); image(Pattern); colormap(Cmap);
axis off; axis square;
