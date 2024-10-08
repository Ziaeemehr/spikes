%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all; hold off; clc; close all;
Total_Neurons = 5; %Solve for this number of interacting Neurons
DT = 2; %Time increment as fraction of time constant
Final_Time = 1000; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
Tau = 20; %Neural time constants in msec
WTS = [1 2 2 1]; %Runge-Kutta Coefficient weights

for NU = 1:Total_Neurons; %Initialize
    X(NU, :) = zeros(1, Last); %Vector to store response of Neuron #1
    K(NU, :) = zeros(1, 4); %Runge-Kutta terms
    X(1, 1) = 0; %Initial conditions here if different from zero
    X(2, 1) = 0; %Initial conditions here if different from zero
    Weights(NU, :) = WTS; %Make into matrix for efficiency in main loop
end;

Wt2 = [0 .5 .5 1]; %Second set of RK weights
rkIndex = [1 1 2 3];

%**********
Stim1 = 80;
Stim2 = 79.8;
Stim3 = 79.8;
Stim4 = 79.8;
Stim5 = 79.8;
%**********

T1 = clock;

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade

        Inh = sum(XH) - XH; %Inhibitory strength
        PSP1 = (Stim1 - 3 * Inh(1)) * (Stim1 > 3 * Inh(1));
        PSP2 = (Stim2 - 3 * Inh(2)) * (Stim2 > 3 * Inh(2));
        PSP3 = (Stim3 - 3 * Inh(3)) * (Stim3 > 3 * Inh(3));
        PSP4 = (Stim4 - 3 * Inh(4)) * (Stim4 > 3 * Inh(4));
        PSP5 = (Stim5 - 3 * Inh(5)) * (Stim5 > 3 * Inh(5));
        K(1, rk) = DT / Tau * (-XH(1) + 100 * (PSP1) ^ 2 / (120 ^ 2 + (PSP1) ^ 2)); %Your Equation Here
        K(2, rk) = DT / Tau * (-XH(2) + 100 * (PSP2) ^ 2 / (120 ^ 2 + (PSP2) ^ 2)); %Your Equation Here
        K(3, rk) = DT / Tau * (-XH(3) + 100 * (PSP3) ^ 2 / (120 ^ 2 + (PSP3) ^ 2)); %Your Equation Here
        K(4, rk) = DT / Tau * (-XH(4) + 100 * (PSP4) ^ 2 / (120 ^ 2 + (PSP4) ^ 2)); %Your Equation Here
        K(5, rk) = DT / Tau * (-XH(5) + 100 * (PSP5) ^ 2 / (120 ^ 2 + (PSP5) ^ 2)); %Your Equation Here

    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6; %Most efficient with weight matrix
end;

Calculation_Time = etime(clock, T1)
whitebg('w');
Xiso = 0:100; %X for Isoclines
Isocline1 = 100 * (Stim1 - 3 * Xiso) .^ 2 ./ (120 ^ 2 + (Stim1 - 3 * Xiso) .^ 2) .* (3 * Xiso < Stim1);
Isocline2 = 100 * (Stim2 - 3 * Xiso) .^ 2 ./ (120 ^ 2 + (Stim2 - 3 * Xiso) .^ 2) .* (3 * Xiso < Stim2);
figure(1); Za = plot(Time, X); set(Za, 'LineWidth', 2)
Peak = max(X(1, :));
Thresh = X(1, :) > 0.95 * Peak;
[Y, Tm] = max(Thresh);
Latency = Time(Tm)
