%This particular variation has VNa = +55, VK = -92, and VL = -50.52
%Solution of Rinzel-Hodgkin-Huxley Equations
%Tempreature of 16.3 deg C
%The resting potential is -70 mV
hold off; clc; clear all; close all;
global IE VX;
DT = 0.02; %Time increment in msec.
Last = 1000; %Last time step
Time = DT * [1:Last]; %Time vector
V = zeros(1, Last); %Vectors to store results
W = zeros(1, Last);
IE = 5; %input('Stimulating current (range 0-200): '); %External Current
whitebg('w');
S = 1.2714;
V(1) = -70; %Put initial conditions here
W(1) = 0.40388; %This is the resting value
TT1 = clock;

for T = 2:Last;
    VV = V(T - 1);
    WW = W(T - 1);
    IEE = IE;

    for RK = 1:2 %Second Order Runge-Kutta
        M = MM(VV); %This and following lines calculate V dependent terms
        Geq = GG(VV); %Winfinity or equilibrium value
        Tau = 5 * exp(- (VV + 60) ^ 2/55 ^ 2) + 1;
        V(T) = V(T - 1) + (RK / 2) * DT * (IEE - 120 * M ^ 3 * (1 - WW) * (VV - 55) - 36 * (WW / S) ^ 4 * (VV + 92) - 0.3 * (VV + 50.528));
        W(T) = W(T - 1) + (RK / 2) * DT * (3 * (Geq - WW) / Tau);
        VV = V(T);
        WW = W(T);
    end;

end;

WS = (W - 0.40388) * 100 - 70; %Scaled to approximate units
TT2 = etime(clock, TT1) %Timing loop
figure(1), Pz = plot(Time, V, '-r'); set(Pz, 'LineWidth', 2);
axis([0, DT * Last, -100, 55]);
xlabel('Time (ms)'); ylabel('V(t)'); title('Rinzel Approximation to Hodgkin-Huxley');
Wnull = (-99:4:61);

S = 1.2714;
U = -Wnull - 70; %for new Rinzel system
AH = 0.07 * exp(U / 20);
BH = (1 + exp((U + 30) / 10)) .^ (-1);
Hh = AH ./ (AH + BH); %Na+ inactiWnullation Wnullariable
AN = 0.01 * (U + 10) ./ (exp((U + 10) / 10) - 1);
BN = 0.125 * exp(U / 80);
Nn = AN ./ (AN + BN);
DW = S * (Nn + S * (1 - Hh)) ./ (1 + S ^ 2);

WW = zeros(1, 23);
Vplot = (-90:6:54);

for j = 1:25,
    VX = Vplot(j);
    WW(j) = fminbnd('Vnull', 0.1, 0.95);
    disp(WW(j))
end;

figure(2), PF = plot(Wnull, DW, '-b', Vplot, WW, '-k', V, W, '-r'); set(PF, 'LineWidth', 2)
axis([-100, 60, 0, 1]); axis square;
xlabel('V'); ylabel('R'); title('Phase Plane, dV/dt = 0 (black), dR/dt = 0 (blue) & limit cycle (red)');
