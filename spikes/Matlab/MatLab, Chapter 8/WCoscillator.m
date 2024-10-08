%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all; hold off; clc; close all;
Total_Equations = 2; %Solve for this number of interacting Neurons
DT = 0.5; %Time increment as fraction of time constant
Final_Time = 400; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
Tau = 5; %Neural time constants in msec
TauI = 10;
WTS = [1 2 2 1]; %Runge-Kutta Coefficient weights

for NU = 1:Total_Equations; %Initialize
    X(NU, :) = zeros(1, Last); %Vector to store response of Neuron #1
    K(NU, :) = zeros(1, 4); %Runge-Kutta terms
    X(1, 1) = 0; %Initial conditions here if different from zero
    X(2, 1) = 0; %Initial conditions here if different from zero
    Weights(NU, :) = WTS; %Make into matrix for efficiency in main loop
end;

Wt2 = [0 .5 .5 1]; %Second set of RK weights
rkIndex = [1 1 2 3];
Stim = input('Stimulus K = ');
T1 = clock;

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade

        PSP1 = Stim + 1.6 * XH(1) - XH(2);
        PSP1 = PSP1 * (PSP1 > 0);
        K(1, rk) = DT / Tau * (-XH(1) + 100 * PSP1 ^ 2 / (30 ^ 2 + PSP1 ^ 2)); %Your Equation Here
        K(2, rk) = DT / TauI * (-XH(2) + 100 * (1.5 * XH(1)) ^ 2 / (30 ^ 2 + (1.5 * XH(1)) ^ 2)); %Your Equation Here

    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6; %Most efficient with weight matrix
end;

Calculation_Time = etime(clock, T1)
whitebg('w');
Xiso = 0:1:99; %X for Isoclines
Isocline1 = 1.6 * Xiso + Stim - 30 * sqrt(Xiso ./ (100 - Xiso));
Isocline2 = 100 * (1.5 * Xiso) .^ 2 ./ (30 ^ 2 + (1.5 * Xiso) .^ 2);
figure(1); Za = plot(Time, X(1, :), 'r-', Time, X(2, :), 'b-'); set(Za, 'LineWidth', 2)
xlabel('Time (ms)'); ylabel('Spike Rates E (red) & I (blue)');
figure(2); Zb = plot(X(1, :), X(2, :), '-r', Xiso, Isocline1, '-k', Xiso, Isocline2, '--k'); set(Zb, 'LineWidth', 2); axis square;
axis([0 100 0 100]);
xlabel('E'); ylabel('I'); title('Phase Plane (blue arrows are trajectory directions)');
hold on;
[XX, Y] = meshgrid(0:5:100);
P1 = Stim + 1.6 * XX - Y;
P1 = P1 .* (P1 > 0);
DX = -XX + 100 * (P1 .^ 2) ./ (30 ^ 2 + P1 .^ 2);
P2 = 1.5 * XX;
DY = -Y + 100 * (P2 .^ 2) ./ (30 ^ 2 + P2 .^ 2);
Scale = 1 ./ sqrt((DX .^ 2 + DY .^ 2));
DX = DX .* Scale;
DY = DY .* Scale;
quiver(XX, Y, DX, DY, 0.75);
hold off;
