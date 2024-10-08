%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all; hold off; clc; close all;
Total_Equations = 4; %Solve for this number of interacting Neurons
DT = 1; %Time increment as fraction of time constant
Final_Time = 1000; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
Tau = 20; %Neural time constants in msec
TauI = 50;
TauD = input('Time constant for delays = ');
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
KK = 350;
T1 = clock;

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade

        PSP = KK - XH(4);
        PSP = PSP * (PSP > 0);
        K(1, rk) = DT / Tau * (-XH(1) + 100 * PSP ^ 2 / (50 ^ 2 + PSP ^ 2)); %Your Equation Here
        K(2, rk) = DT / TauI * (-XH(2) + 6 * XH(3)); %Your Equation Here
        K(3, rk) = DT / TauD * (-XH(3) + XH(1));
        K(4, rk) = DT / TauD * (-XH(4) + XH(2));

    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6; %Most efficient with weight matrix
end;

TrueSol = 2 * Time .* exp(-Time / 20);
Calculation_Time = etime(clock, T1)
whitebg('w');
figure(1); Za = plot(Time, X(1, :), 'r'); set(Za, 'LineWidth', 2);
xlabel('Time(ms)'); ylabel('E(t)');
figure(2), SSp = plot(X(1, :), X(2, :), 'r-'); set(SSp, 'LineWidth', 2);
xlabel('E'); ylabel('I'); title('E-I Projection of State Space');
