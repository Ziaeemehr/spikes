%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all; hold off; close all;
whitebg('w');
Total_Neurons = 4; %Solve for this number of interacting Neurons
DT = 0.04; %Time increment
Final_Time = 20000; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
Tau = 0.97; %Neural time constants in msec
TauR = 2.1;
TauX = 15; %Tau for Ca++ entry
TauC = 56; %Tau for Iahp potential
WTS = [1 2 2 1]; %Runge-Kutta Coefficient weights

for NU = 1:Total_Neurons; %Initialize
    X(NU, :) = zeros(1, Last); %Vector to store response of Neuron #1
    K(NU, :) = zeros(1, 4); %Runge-Kutta terms
    Weights(NU, :) = WTS; %Make into matrix for efficiency in main loop
end;

X(1, 1) = -0.754; %Initial conditions here if different from zero
X(2, 1) = 0.279; %Initial conditions here if different from zero
Wt2 = [0 .5 .5 1]; %Second set of RK weights
rkIndex = [1 1 2 3];
Stim = input('Stimulating current strength (0-2): ');
T1 = clock;
ST = 10.6;

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade

        K(1, rk) = DT / Tau * (- (17.81 + 47.58 * XH(1) + 33.8 * XH(1) ^ 2) * (XH(1) - 0.48) - 26 * XH(2) * (XH(1) + 0.95) - 1.7 * XH(3) * (XH(1) - 1.4) - 13 * XH(4) * (XH(1) + 0.95) + Stim);
        K(2, rk) = DT / TauR * (-XH(2) + 1.29 * XH(1) + 0.79 + 3.3 * (XH(1) + 0.38) ^ 2);
        K(3, rk) = DT / TauX * (-XH(3) + 9 * (XH(1) + 0.754) * (XH(1) + 0.7));
        K(4, rk) = DT / TauC * (-XH(4) + 3 * XH(3));

    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6;
end;

Calculation_Time = etime(clock, T1)
figure(1), ZA = plot(Time(2500:Last) - 100, 100 * X(1, 2500:Last), 'r-'); set(ZA, 'LineWidth', 2);
axis([0, 500, -80, 20]);
xlabel('Time (ms)'); ylabel('Potential (mV)');
VV = -0.9:0.01:1.5;
DVdt = -0.5 * ((1.37 + 3.66 * VV + 2.6 * VV .^ 2) .* (VV - 0.48) - Stim / 13) ./ (VV + 0.95);
DRdt = 1.29 * VV + 0.79 + 3.3 * (VV + 0.38) .^ 2;
figure(2), ZB = plot(VV, DVdt, 'k-', VV, DRdt, 'b-', X(1, :), X(2, :), 'r-'); axis([-1, 0.6, 0, 1]);
set(ZB, 'LineWidth', 2); axis square;
Spikes = (X(1, 1:Last - 1) < -0.12) .* (X(1, 2:Last) >= -0.12);
SpkTime = zeros(1, sum(Spikes));
Nspk = 1; %Number of spike

for T = 1:length(Spikes); %Calculate spike rate for all interspike intervals
    if Spikes(T) == 1; SpkTime(Nspk) = T * DT; Nspk = Nspk + 1; end;
end;

Final = length(SpkTime);
Rates = 1000 ./ (SpkTime(2:Final) - SpkTime(1:Final - 1));
Rates'
figure(3), FFf = plot(X(4, round(Last / 2):Last), X(3, round(Last / 2):Last), 'r-'); set(FFf, 'LineWidth', 2);
xlabel('C'); ylabel('X'); title('X-C Projection of Phase Space');
