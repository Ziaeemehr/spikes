%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all; hold off; clc; close all;
Total_Equations = 2; %Solve for this number of interacting Neurons
DT = 0.005; %Time increment as fraction of time constant
Final_Time = 10; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
Tau = 0.1; %Neural time constants in msec
TauR = 1.25
WTS = [1 2 2 1]; %Runge-Kutta Coefficient weights

for NU = 1:Total_Equations; %Initialize
    X(NU, :) = zeros(1, Last); %Vector to store response of Neuron #1
    K(NU, :) = zeros(1, 4); %Runge-Kutta terms
    X(1, 1) = -1.5; %Initial conditions here if different from zero
    X(2, 1) = -3/8; %Initial conditions here if different from zero
    Weights(NU, :) = WTS; %Make into matrix for efficiency in main loop
end;

Wt2 = [0 .5 .5 1]; %Second set of RK weights
rkIndex = [1 1 2 3];
Input = input('Stimulus strength = ');
T1 = clock;

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade

        K(1, rk) = DT / Tau * (XH(1) - XH(1) ^ 3/3 - XH(2) + Input); %Your Equation Here
        K(2, rk) = DT / TauR * (-XH(2) + 1.25 * XH(1) + 1.5); %Your Equation Here

    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6; %Most efficient with weight matrix
end;

disp(X(:, 1:5))
Calculation_Time = etime(clock, T1)
whitebg('w');
Xiso = -3.2:0.01:3.2; %X for Isoclines
Isocline1 = Xiso - Xiso .^ 3/3 + Input;
Isocline2 = Xiso / 0.8 + 1.5;
figure(1); Za = plot(Time, X(1, :), 'r-', Time, X(2, :), 'b-'); set(Za, 'LineWidth', 2)
ylabel('V(t) (red) & R(t) (blue'); xlabel('Time(ms)');
figure(2); Zb = plot(X(1, :), X(2, :), '-r', Xiso, Isocline1, '-k', Xiso, Isocline2, '--k'); set(Zb, 'LineWidth', 2); axis square;
axis([-3.5 3.5 -6 8]); xlabel('V'); ylabel('R');
VMax = max(X(1, Last / 2:Last))
