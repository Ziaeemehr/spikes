%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all; hold off; close all; clc;
Total_Equations = 2; %Solve for this number of interacting Neurons
DT = 0.04; %Time increment
Final_Time = 100; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
Tau = 0.97; %Neural time constants in msec
TauR = 5.0;
WTS = [1 2 2 1]; %Runge-Kutta Coefficient weights

for NU = 1:Total_Equations; %Initialize
    X(NU, :) = zeros(1, Last); %Vector to store response of Neuron #1
    K(NU, :) = zeros(1, 4); %Runge-Kutta terms
    Weights(NU, :) = WTS; %Make into matrix for efficiency in main loop
end;

X(1, 1) = -0.5; %Initial conditions here if different from zero
X(2, 1) = 0; %Initial conditions here if different from zero
Wt2 = [0 .5 .5 1]; %Second set of RK weights
rkIndex = [1 1 2 3];
Stim = input('Stimulating current strength (0-2): ');
whitebg('w');
T1 = clock;
ST = 10.6;

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade

        TauR = 1 / cosh((XH(1) - 0.1) / 0.14);

        %TauR = 0.5; %Remove comment here to use constant tau

        K(1, rk) = DT / Tau * (-1 / (1 + exp(- (XH(1) + 0.01) / 0.075)) * (XH(1) - 1.0) - 2 * XH(2) * (XH(1) + 0.7) - 0.5 * (XH(1) + 0.5) + Stim);
        K(2, rk) = DT * 0.2 / TauR * (-XH(2) + 1 / (1 + exp(- (XH(1) - 0.1) / 0.07)));

    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6;
end;

Calculation_Time = etime(clock, T1)
figure(1), ZA = plot(Time, X(1, :), 'r-'); set(ZA, 'LineWidth', 2);
xlabel('Time (ms)'); ylabel('Potential (mV)');
VV = -0.69:0.01:0.7;
DVdt = 0.5 * (- (VV - 1) ./ (1 + exp(- (VV + 0.01) / 0.075)) - 0.5 * (VV + 0.5) + Stim) ./ (VV + 0.7);
DRdt = 1 ./ (1 + exp(- (VV - 0.1) / 0.07));
figure(2), ZB = plot(VV, DVdt, 'k-', VV, DRdt, 'b-', X(1, :), X(2, :), 'r-'); axis([-0.8, 0.7, -0.2, 0.5]);
set(ZB, 'LineWidth', 2); axis square;
title('dV/dt = 0 isocline (black), dR/dt = 0 isocline (blue) & limit cycle (red');
Spikes = (X(1, 1:Last - 1) < -0.12) .* (X(1, 2:Last) >= -0.12);
SpkTime = zeros(1, sum(Spikes));
Nspk = 1; %Number of spike

for T = 1:length(Spikes); %Calculate spike rate for all interspike intervals
    if Spikes(T) == 1; SpkTime(Nspk) = T * DT; Nspk = Nspk + 1; end;
end;

Final = length(SpkTime);
Rates = 1000 ./ (SpkTime(2:Final) - SpkTime(1:Final - 1));
Spike_Rate = mean(Rates)
