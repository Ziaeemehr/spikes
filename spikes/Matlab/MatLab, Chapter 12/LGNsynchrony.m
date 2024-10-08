%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all; hold off;
global DT Tau TauR TauC TauH ES IS II TauSyn Tme SynThresh;
whitebg('w');
Total_Neurons = 28; %Solve for this number of interacting Neurons
DT = 0.1; %Time increment
Final_Time = 1100; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
Tau = 0.97; %Neural time constants in msec
TauR = 5.6;
TauC = 30; %Tau for Ca++ entry
TauH = 100; %Tau for Iahp potential
WTS = [1 2 2 1]; %Runge-Kutta Coefficient weights

for NU = 1:Total_Neurons; %Initialize
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
X(15, 1) = -0.736; %Initial conditions here if different from zero
X(16, 1) = 0.258; %Initial conditions here if different from zero
X(17, 1) = 0.095; %Initial conditions here if different from zero
X(18, 1) = 0.323; %Initial conditions here if different from zero
X(19, 1) = -0.736; %Initial conditions here if different from zero
X(20, 1) = 0.258; %Initial conditions here if different from zero
X(21, 1) = 0.095; %Initial conditions here if different from zero
X(22, 1) = 0.323; %Initial conditions here if different from zero
Wt2 = [0 .5 .5 1]; %Second set of RK weights
rkIndex = [1 1 2 3];
SynThresh = -0.2; %Threshold for IPSP conductance change
ES = input('Excitatory synaptic conductance factor (0-6): ');
IS = input('Inhibitory synaptic conductance factor: ');
II = input('Inhibitory-Inhibitory synaptic conductance factor: ');

%**********
TauSyn = 40; %IPSP time constant
%**********

Stim = 0; %Endodginous burster
T1 = clock;

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade

        K(1:14, rk) = LGNoscillator(XH(1:14), 0.25, XH(19));
        K(15:28, rk) = LGNoscillator(XH(15:28), 0, XH(5));

        if rem(Tme, 100) == 0;
            figure(1), ZA = plot(Time, 100 * X(1, :), 'r-', Time, 100 * X(5, :), 'b-', Time, 100 * X(15, :) - 100, 'r-', Time, 100 * X(19, :) - 100, 'b-'); %set(ZA,'LineWidth', 2);
            xlabel('Time (ms)'); ylabel('Potential (mV)');
            pause(0.1);
        end;

    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6;
end;

Calculation_Time = etime(clock, T1)
figure(1), ZA = plot(Time, 100 * X(1, :), 'r-', Time, 100 * X(5, :), 'b-', Time, 100 * X(15, :) - 100, 'r-', Time, 100 * X(19, :) - 100, 'b-'); %set(ZA,'LineWidth', 2);
xlabel('Time (ms)'); ylabel('Potential (mV)');
[Spike_Height, Tm] = max(X(1, :));
Half_Ht = (Spike_Height - X(1, 1)) / 2 + X(1, 1);
[Value, Up] = max(X(1, 1:Tm) >= Half_Ht);
[Value, Down] = max(X(1, Tm:Last) <= Half_Ht);
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
