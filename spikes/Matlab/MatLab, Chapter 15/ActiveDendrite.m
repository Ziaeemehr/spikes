%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all; hold off;
whitebg('w');
Total_Equations = 8; %Solve for this number of interacting Neurons
DT = 0.04; %Time increment
Final_Time = 30; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
Tau = 0.97; %Neural time constants in msec
TauR = 5.6;
WTS = [1 2 2 1]; %Runge-Kutta Coefficient weights

for NU = 1:Total_Equations; %Initialize
    X(NU, :) = zeros(1, Last); %Vector to store response of Neuron #1
    K(NU, :) = zeros(1, 4); %Runge-Kutta terms
    Weights(NU, :) = WTS; %Make into matrix for efficiency in main loop
end;

Stimulus = zeros(1, Last);
X(1, 1) = -0.754; %Initial conditions here if different from zero
X(2, 1) = 0.279; %Initial conditions here if different from zero
X(3, 1) = -0.754; %Initial conditions here if different from zero
X(4, 1) = 0.279; %Initial conditions here if different from zero
X(5, 1) = -0.754; %Initial conditions here if different from zero
X(6, 1) = 0.279; %Initial conditions here if different from zero
X(7, 1) = -0.754; %Initial conditions here if different from zero
X(8, 1) = 0.279; %Initial conditions here if different from zero
Wt2 = [0 .5 .5 1]; %Second set of RK weights
rkIndex = [1 1 2 3];
SorD = input('Stimulate (1) Soma or (2) Apical Dendrite: ');
Stim = 0.45;
T1 = clock;

%**********
gc = 4; %Conductance between compartments
NaX = 0.05;
%**********

ST = 10.6;

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade
        ST = Stim * (Tme >= 2) * (Tme <= 25);

        K(1, rk) = DT / Tau * (- (17.81 + 47.58 * XH(1) + 33.8 * XH(1) ^ 2) * (XH(1) - 0.48) - 26 * XH(2) * (XH(1) + 0.95) + gc * (XH(3) - XH(1)) + ST * (SorD == 1));
        K(2, rk) = DT / TauR * (-XH(2) + 1.29 * XH(1) + 0.79 + 3.3 * (XH(1) + 0.38) ^ 2);
        K(3, rk) = DT / Tau * (NaX * (- (17.81 + 47.58 * XH(3) + 33.8 * XH(3) ^ 2) * (XH(3) - 0.48) - 26 * XH(4) * (XH(3) + 0.95)) + gc * (XH(1) - XH(3)) + gc * (XH(5) - XH(3)));
        K(4, rk) = DT / TauR * (-XH(4) + 1.29 * XH(3) + 0.79 + 3.3 * (XH(3) + 0.38) ^ 2);
        K(5, rk) = DT / Tau * (NaX * (- (17.81 + 47.58 * XH(5) + 33.8 * XH(5) ^ 2) * (XH(5) - 0.48) - 26 * XH(6) * (XH(5) + 0.95)) + gc * (XH(3) - XH(5)) + gc * (XH(7) - XH(5)));
        K(6, rk) = DT / TauR * (-XH(6) + 1.29 * XH(5) + 0.79 + 3.3 * (XH(5) + 0.38) ^ 2);
        K(7, rk) = DT / Tau * (NaX * (- (17.81 + 47.58 * XH(7) + 33.8 * XH(7) ^ 2) * (XH(7) - 0.48) - 26 * XH(8) * (XH(7) + 0.95)) + gc * (XH(5) - XH(7)) + ST * (SorD == 2));
        K(8, rk) = DT / TauR * (-XH(8) + 1.29 * XH(7) + 0.79 + 3.3 * (XH(7) + 0.38) ^ 2);

    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6;
    Stimulus(T) = ST;
end;

Calculation_Time = etime(clock, T1)
figure(1), ZA = plot(Time, 100 * X(1, :), 'r-', Time, 100 * X(7, :), 'b-'); set(ZA, 'LineWidth', 2);
xlabel('Time (ms)'); ylabel('Potential (mV)');
title('Soma Potential (red) and Proximal Dendritic Potential (blue)');
VV = -0.9:0.01:1.5;
DVdt = -0.5 * ((1.37 + 3.66 * VV + 2.6 * VV .^ 2) .* (VV - 0.48) - Stim / 13) ./ (VV + 0.95);
DRdt = 1.29 * VV + 0.79 + 3.3 * (VV + 0.38) .^ 2;
figure(2), ZB = plot(VV, DVdt, 'k-', VV, DRdt, 'b-', X(1, :), X(2, :), 'r-'); axis([-1, 0.6, 0, 1]);
set(ZB, 'LineWidth', 2); axis square;
[Soma, Stime] = max(X(1, :));
[Dend, Dtime] = max(X(7, :));
Soma_Spike = 100 * (Soma + 0.754)
Dendrite_Spike = 100 * (Dend + 0.754)
Time_Diff = Time(Dtime) - Time(Stime)
disp(X(1:30))
