%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all; hold off; close all; clc;
Total_Neurons = 2; %Solve for this number of interacting Neurons
DT = 0.02; %Time increment as fraction of time constant
Final_Time = 20; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
Tau = 0.8; %Neural time constants in msec
TauR = 1.9;
WTS = [1 2 2 1]; %Runge-Kutta Coefficient weights

for NU = 1:Total_Neurons; %Initialize
    X(NU, :) = zeros(1, Last); %Vector to store response of Neuron #1
    K(NU, :) = zeros(1, 4); %Runge-Kutta terms
    Weights(NU, :) = WTS; %Make into matrix for efficiency in main loop
end;

X(1, 1) = -0.70; %Initial conditions here if different from zero
X(2, 1) = 0.088; %Initial conditions here if different from zero
Wt2 = [0 .5 .5 1]; %Second set of RK weights
rkIndex = [1 1 2 3];
Stim = input('Stimulating current strength (0-2): ');
whitebg('w');
T1 = clock;
ST = 10.6;

for T = 2:3 %:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade

        K(1, rk) = DT / Tau * (- (17.81 + 47.71 * XH(1) + 32.63 * XH(1) ^ 2) * (XH(1) - 0.55) - 26 * XH(2) * (XH(1) + 0.92) + Stim);
        K(2, rk) = DT / TauR * (-XH(2) + 1.35 * XH(1) + 1.03);
        disp(XH)
        disp(K)
    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6;
    disp('X(:, T)')
    disp(X(:, T))
end;

Calculation_Time = etime(clock, T1)
figure(1), ZA = plot(Time, 100 * X(1, :), 'r-'); set(ZA, 'LineWidth', 2)
VV = -0.9:0.01:1.5;
DVdt = -0.5 * ((1.37 + 3.67 * VV + 2.51 * VV .^ 2) .* (VV - 0.55) - Stim / 13) ./ (VV + 0.92);
DRdt = 1.35 * VV + 1.03;
figure(2), ZB = plot(VV, DVdt, 'k-', VV, DRdt, 'b-', X(1, :), X(2, :), 'r-'); axis([-1, 0.6, 0, 1]);
set(ZB, 'LineWidth', 2); axis square;
Spike_Height = max(X(1, round(Last / 2):Last))
