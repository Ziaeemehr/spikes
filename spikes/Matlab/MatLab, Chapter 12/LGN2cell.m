%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all; hold off;
whitebg('w');
Total_Equations = 12; %Solve for this number of interacting Neurons
DT = 0.1; %Time increment
Final_Time = 1000; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
Tau = 0.97; %Neural time constants in msec
TauR = 5.6;
TauC = 30; %Tau for Ca++ entry
TauH = 100; %Tau for Iahp potential
WTS = [1 2 2 1]; %Runge-Kutta Coefficient weights

for NU = 1:Total_Equations; %Initialize
    X(NU, :) = zeros(1, Last); %Vector to store response of Neuron #1
    K(NU, :) = zeros(1, 4); %Runge-Kutta terms
    Weights(NU, :) = WTS; %Make into matrix for efficiency in main loop
end;

X(1, 1) = -0.736; %Initial conditions here if different from zero
X(2, 1) = 0.258; %Initial conditions here if different from zero
X(3, 1) = 0.095; %Initial conditions here if different from zero
X(4, 1) = 0.323; %Initial conditions here if different from zero
X(5, 1) = -0.736; %Initial conditions here if different from zero
X(6, 1) = 0.258; %Initial conditions here if different from zero
X(7, 1) = 0.095; %Initial conditions here if different from zero
X(8, 1) = 0.323; %Initial conditions here if different from zero
Wt2 = [0 .5 .5 1]; %Second set of RK weights
rkIndex = [1 1 2 3];
SynThresh = -0.2; %Threshold for IPSP conductance change
ES = input('Excitatory synaptic conductance factor (0-6): ');
IS = input('Inhibitory synaptic conductance factor: ');

%**********
TauSyn = 40; %IPSP time constant
%**********

Stim = 1; %Endodginous burster
T1 = clock;

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade

        K(1, rk) = DT / Tau * (- (17.81 + 47.58 * XH(1) + 33.8 * XH(1) ^ 2) * (XH(1) - 0.48) - 26 * XH(2) * (XH(1) + 0.95) - 1.93 * XH(3) * (1 - 0.5 * XH(4)) * (XH(1) - 1.4) - 3.25 * XH(4) * (XH(1) + 0.95) - Stim * (Tme >= 20) * (Tme <= 70) - IS * XH(10) * (XH(1) + 0.92));
        K(2, rk) = DT / TauR * (-XH(2) + 1.29 * XH(1) + 0.79 + 3.3 * (XH(1) + 0.38) ^ 2);
        K(3, rk) = DT / TauC * (-XH(3) + 6.65 * (XH(1) + 0.86) * (XH(1) + 0.84));
        K(4, rk) = DT / TauH * (-XH(4) + 3.0 * XH(3));
        K(9, rk) = DT / TauSyn * (-XH(9) + (XH(5) > SynThresh));
        K(10, rk) = DT / TauSyn * (-XH(10) + XH(9));

        K(5, rk) = DT / Tau * (- (17.81 + 47.58 * XH(5) + 33.8 * XH(5) ^ 2) * (XH(5) - 0.48) - 26 * XH(6) * (XH(5) + 0.95) - 1.93 * XH(7) * (1 - 0.5 * XH(8)) * (XH(5) - 1.4) - 3.25 * XH(8) * (XH(5) + 0.95) - ES * XH(12) * (XH(5)));
        K(6, rk) = DT / TauR * (-XH(6) + 1.29 * XH(5) + 0.79 + 3.3 * (XH(5) + 0.38) ^ 2);
        K(7, rk) = DT / TauC * (-XH(7) + 6.65 * (XH(5) + 0.86) * (XH(5) + 0.84));
        K(8, rk) = DT / TauH * (-XH(8) + 3.0 * XH(7));
        K(11, rk) = DT / TauSyn * (-XH(11) + (XH(1) > SynThresh));
        K(12, rk) = DT / TauSyn * (-XH(12) + XH(11));

        if rem(Tme, 100) == 0;
            figure(1), ZA = plot(Time, 100 * X(1, :), 'r-', Time, 100 * X(5, :), 'b-'); %set(ZA,'LineWidth', 2);
            xlabel('Time (ms)'); ylabel('Potential (mV)');
            pause(0.1);
        end;

    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6;
end;

Calculation_Time = etime(clock, T1)
figure(1), ZA = plot(Time, 100 * X(1, :), 'r-', Time, 100 * X(5, :), 'b-'); set(ZA, 'LineWidth', 2);
xlabel('Time (ms)'); ylabel('Potential (mV)');
[Spike_Height, Tm] = max(X(1, :));
Half_Ht = (Spike_Height - X(1, 1)) / 2 + X(1, 1);
[Value, Up] = max(X(1, 1:Tm) >= Half_Ht);
[Value, Down] = max(X(1, Tm:Last) <= Half_Ht);
Spike_Width_ms = (Down + Tm - Up) * DT
%PotentialChange = 100*(min(X(1, Last/2:Last)) - X(1, 1))
Spikes = (X(1, 1:Last - 1) < -0.12) .* (X(1, 2:Last) >= -0.12);
SpkTime = zeros(1, sum(Spikes));
Nspk = 1; %Number of spike

for T = 1:length(Spikes); %Calculate spike rate for all interspike intervals
    if Spikes(T) == 1; SpkTime(Nspk) = T * DT; Nspk = Nspk + 1; end;
end;

Final = length(SpkTime);
Rates = 1000 ./ (SpkTime(2:Final) - SpkTime(1:Final - 1));
Rates'
