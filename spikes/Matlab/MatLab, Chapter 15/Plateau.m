%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all; hold off;
Total_Equations = 4; %Solve for this number of interacting Neurons
DT = 0.08; %Time increment
Final_Time = 1800; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
Tau = 0.97; %Neural time constants in msec
TauR = 5.6;
TauC = 2; %Tau for Ca++ entry
TauH = 20; %Tau for Iahp potential
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
Wt2 = [0 .5 .5 1]; %Second set of RK weights
rkIndex = [1 1 2 3];
Stim = input('Stimulating current strength (0-2): ');
whitebg('w');
T1 = clock;

%**********
gc = 0.1; %Conductance between dendrite and soma
p = 0.37; %Proportion of total cell area in soma
GAHP = 1;
%**********

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade
        ST = Stim * (Tme >= 40) * (Tme < 117) - Stim * (Tme >= 1450) * (Tme < 1525);

        K(1, rk) = DT / Tau * (- (17.81 + 47.58 * XH(1) + 33.8 * XH(1) ^ 2) * (XH(1) - 0.48) - 26 * XH(2) * (XH(1) + 0.95) + gc / p * (XH(3) - XH(1)) + ST);
        K(2, rk) = DT / TauR * (-XH(2) + 1.29 * XH(1) + 0.79 + 3.3 * (XH(1) + 0.38) ^ 2);
        K(3, rk) = DT / TauC * (- (XH(3) + 0.754) * (XH(3) + 0.7) * (XH(3) - 1.0) - GAHP * XH(4) * (XH(3) + 0.95) + gc / (1 - p) * (XH(1) - XH(3)));
        K(4, rk) = DT / TauH * (-XH(4) + 0.5 * (XH(3) + 0.754));

    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6;
    Stimulus(T) = ST;
end;

Calculation_Time = etime(clock, T1)
%figure(1), ZA = plot(Time, 100*X(1, :), 'r-', Time, 100*X(3,  :) - 200, 'b-', Time, Stimulus*10 - 100, 'b-'); set(ZA, 'LineWidth', 2);
figure(1), ZA = plot(Time, 100 * X(1, :), 'r-', Time, Stimulus * 10 - 100, 'b-'); set(ZA, 'LineWidth', 2);
xlabel('Time (ms)'); ylabel('Potential (mV)');
title('Soma Voltage (red) & Dendritic Plateau Potential (blue)');
VV = -0.9:0.01:1;
DVdt = (1 / GAHP) * (- (VV + 0.754) .* (VV + 0.7) .* (VV - 1) + gc / (1 - p) * (-VV - 0.754)) ./ (VV + 0.95);
DRdt = 0.5 * (VV + 0.754);
figure(2), ZB = plot(VV, DVdt, 'k-', VV, DRdt, 'b-', X(3, :), X(4, :), 'r-'); axis([-1, 1, -0.1, 1]);
set(ZB, 'LineWidth', 2); axis square; title('Dendritic Phase Plane'); xlabel('VD'); ylabel('C');
Spikes = (X(1, 1:Last - 1) < -0.12) .* (X(1, 2:Last) >= -0.12);
SpkTime = zeros(1, sum(Spikes));
Nspk = 1; %Number of spike

for T = 1:length(Spikes); %Calculate spike rate for all interspike intervals
    if Spikes(T) == 1; SpkTime(Nspk) = T * DT; Nspk = Nspk + 1; end;
end;

Final = length(SpkTime);
Spike_Rates = 1000 ./ (SpkTime(2:Final) - SpkTime(1:Final - 1))'
