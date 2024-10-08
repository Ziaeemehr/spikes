%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all; hold off; clc; close all;
Total_Equations = 2; %Solve for this number of interacting Neurons
DT = 0.02; %Time increment as fraction of time constant
Final_Time = 20; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
Tau = 20; %Neural time constants in msec
Beta = input('Hopf parameter Beta = ');
WTS = [1 2 2 1]; %Runge-Kutta Coefficient weights

for NU = 1:Total_Equations; %Initialize
    X(NU, :) = zeros(1, Last); %Vector to store response of Neuron #1
    K(NU, :) = zeros(1, 4); %Runge-Kutta terms
    X(1, 1) = 1; %Initial conditions here if different from zero
    X(2, 1) = 0; %Initial conditions here if different from zero
    Weights(NU, :) = WTS; %Make into matrix for efficiency in main loop
end;

Wt2 = [0 .5 .5 1]; %Second set of RK weights
rkIndex = [1 1 2 3];
Stim = 0;
T1 = clock;

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade

        K(1, rk) = DT * (XH(2)); %Your Equation Here
        K(2, rk) = DT * (XH(2) * (Beta - XH(1) ^ 2) - 25 * XH(1)); %Your Equation Here
    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6; %Most efficient with weight matrix
end;

Calculation_Time = etime(clock, T1)
whitebg('w');
figure(1); Za = plot(Time, X(1, :), 'r', Time, X(2, :), '-k'); set(Za, 'LineWidth', 2)
xlabel('Time'); ylabel('X (red) & Y (blue)'); title('Van der Pol Equation');
figure(2), SSp = plot(X(1, :), X(2, :), 'r-'); set(SSp, 'LineWidth', 2);
xlabel('X'); ylabel('Y'); axis square;
hold on;
MX = max(X(1, :)); MN = min(X(1, :));
XX = 1.5 * MN:1.5 * (MX - MN) / 20:1.5 * MX;
YMX = max(X(2, :)); YMN = min(X(2, :));
YY = 1.5 * YMN:1.5 * (YMX - YMN) / 20:1.5 * YMX;
[X, Y] = meshgrid(XX, YY);
DX = Y;
DY = -25 * X + Y .* (Beta - X .^ 2);
Scale = 1 ./ sqrt((DX .^ 2 + DY .^ 2));
DX = DX .* Scale;
DY = DY .* Scale;
quiver(X, Y, DX, DY, 1/2);
hold off;
