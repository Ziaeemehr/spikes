%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all; hold off; clc; close all;
Total_Equations = 4; %Solve for this number of interacting Neurons
DT = 0.1; %Time increment as fraction of time constant
Final_Time = 600; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
TauR = 5;
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
Input = input('Stimulus strength = ');

G = 0.047;
P = 0.48;

T1 = clock;

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade

        K(1, rk) = DT * (-4 * (XH(1) ^ 2 - XH(1) / 10) * (XH(1) - 1) - XH(2) * (XH(1) +1/5) - XH(3) * (XH(1) - 1) - XH(4) * (XH(1) +1/5) + Input); %Your Equation Here
        K(2, rk) = DT / TauR * (-XH(2) + 3 * XH(1) ^ 2); %Your Equation Here
        K(3, rk) = DT / 40 * (-XH(3) + 2 * XH(1) ^ 2);
        K(4, rk) = DT / 50 * (-XH(4) + 4 * XH(3));

    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6; %Most efficient with weight matrix
end;

Calculation_Time = etime(clock, T1)
whitebg('w');
Xiso = -0.19:0.01:1; %X for Isoclines
Isocline1 = (-4 * (Xiso .^ 2 - Xiso / 10) .* (Xiso - 1) + Input) ./ (Xiso +1/5); ;
Isocline2 = 3 * Xiso .^ 2;
figure(1); Za = plot(Time, X(1, :), 'r-'); set(Za, 'LineWidth', 2)
ylabel('V (Normalized Units)'); xlabel('Time (arbitrary units)');
figure(2); Zb = plot(X(1, :), X(2, :), '-r', Xiso, Isocline1, '-k', Xiso, Isocline2, '--k'); set(Zb, 'LineWidth', 2); axis square;
axis([-0.2 1 -0.2 1]); xlabel('V'); ylabel('R');
