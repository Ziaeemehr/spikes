%Isocline movie of neural adaptation in short term memory
Xiso = 0:100; %X for Isoclines
A = 0;
Isocline1 = 100 * (3 * Xiso) .^ 2 ./ ((120 + 0.7 * A) ^ 2 + (3 * Xiso) .^ 2);
Isocline2 = 100 * (3 * Xiso) .^ 2 ./ ((120 + 0.7 * A) ^ 2 + (3 * Xiso) .^ 2);
figure(1); Zb = plot(Xiso, Isocline1, '-r', Isocline2, Xiso, '-k'); set(Zb, 'LineWidth', 2); axis square;

for K = 1:20;
    A = A + 2;
    Isocline1 = 100 * (3 * Xiso) .^ 2 ./ ((120 + A) ^ 2 + (3 * Xiso) .^ 2);
    Isocline2 = 100 * (3 * Xiso) .^ 2 ./ ((120 + A) ^ 2 + (3 * Xiso) .^ 2);
    figure(1); Zb = plot(Xiso, Isocline1, '-r', Isocline2, Xiso, '-k'); set(Zb, 'LineWidth', 2); axis square;
    pause(0.1);
end;
