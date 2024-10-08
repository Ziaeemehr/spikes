%Movie of simulated Lamprey swimming
%Mechanics are simplified
whitebg('w');
clear all;
Seg = 0.3; %Length per segment in cm, 30 cm total
Spine = (1:100);
Width = (Spine > 70) + Spine .* (Spine <= 70) / 70 + 0.05;
max(Width)
Spine = Seg * Spine;
Mass = Width + 0.05;
Width(97:100) = [1, 0.8, 0.5, .1]; %Make a nice head
Hz = 2.3;
V = Hz * 30; %traveling wave velocity in cm/sec
T = 0; %time, to be run in loop
DT = 0.03;
K = 0.034;

for T = 0:1:40;
    Neurons = 0.83 * sin(2 * pi * (Spine + V * T * DT) / 30 + 0.2) + 0.17 * sin(6 * pi * (Spine + V * T * DT) / 30 - 1.53);
    SegAngle = asin((K / Seg) * Neurons ./ Mass);
    LampreyY = zeros(1, 100);
    LampreyX = zeros(1, 100);
    LampreyX(100) = Seg * cos(SegAngle(100));
    LampreyY(100) = Seg * sin(SegAngle(100));

    for SS = 99:-1:1; %compute total angle of segment
        LampreyX(SS) = LampreyX(SS + 1) + Seg * cos(SegAngle(SS));
        LampreyY(SS) = LampreyY(SS + 1) + Seg * sin(SegAngle(SS));
    end;

    LampreyY = LampreyY - LampreyY(95);
    LampreyTop = fliplr(LampreyY + Width);
    LampreyBottom = fliplr(LampreyY - Width);
    LampreyX = LampreyX + 0.8 * V * T * DT;
    Head(T + 1) = LampreyY(100);
    Middle(T + 1) = LampreyY(30);
    Tail(T + 1) = LampreyY(1);
    figure(1), P1 = plot(LampreyX, LampreyTop, 'r-', LampreyX, LampreyBottom, 'r-'); axis equal;
    axis([0, 100, -20, 20]);
    set(P1, 'LineWidth', 2);
    pause(0.1);
end;
