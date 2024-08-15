%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all; hold off; clc;
Total_Equations = 9; %Solve for this number of interacting Neurons
DT = 2; %Time increment as fraction of time constant
Final_Time = 900; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
WTS = [1 2 2 1]; %Runge-Kutta Coefficient weights

for NU = 1:Total_Equations; %Initialize
    X(NU, :) = zeros(1, Last); %Vector to store response of Neuron #1
    K(NU, :) = zeros(1, 4); %Runge-Kutta terms
    X(1, 1) = 2; %Initial conditions to create slight left-right imbalance
    Weights(NU, :) = WTS; %Make into matrix for efficiency in main loop
end;

Wt2 = [0 .5 .5 1]; %Second set of RK weights
rkIndex = [1 1 2 3];

%**********
Stim = input('Strength of stimulus = ');
HT5 = 1; %Change to zero for blocked 5-HT modulation
Ginhib = 1; %Strength of inhibitory cross-coupling, change to zero for independence
EE = 6; % E to E synaptic strength
%**********

whitebg('w');
Tau = 9; %Neural time constants in msec
TauC = 9; %Neural time constants in msec
TauHT = 12; %Time constant for serotonin neuron to have its effect.
EC = 2; % E to C synaptic strength
T1 = clock;

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade

        TauA = 400 / (1 + HT5 * (0.2 * XH(9)) ^ 2); %Effects of stimulus on tau via 5-HT
        AHPgain = 6 + HT5 * (0.09 * XH(9)) ^ 2;
        PSP1 = XH(9) + EE * XH(1) - Ginhib * XH(6);
        PSP1 = PSP1 * (PSP1 > 0);
        PSP2 = XH(9) + EE * XH(2) - Ginhib * XH(5);
        PSP2 = PSP2 * (PSP2 > 0);
        K(1, rk) = DT / Tau * (-XH(1) + 100 * (PSP1) ^ 2 / ((64 + AHPgain * XH(3)) ^ 2 + (PSP1) ^ 2)); %Your Equation Here
        K(2, rk) = DT / Tau * (-XH(2) + 100 * (PSP2) ^ 2 / ((64 + AHPgain * XH(4)) ^ 2 + (PSP2) ^ 2)); %Your Equation Here
        K(3, rk) = DT / TauA * (-XH(3) + XH(1));
        K(4, rk) = DT / TauA * (-XH(4) + XH(2));

        PSP5 = XH(9) + EC * XH(1) - Ginhib * XH(6);
        PSP5 = PSP5 * (PSP5 > 0);
        PSP6 = XH(9) + EC * XH(2) - Ginhib * XH(5);
        PSP6 = PSP6 * (PSP6 > 0);
        K(5, rk) = DT / TauC * (-XH(5) + 100 * (PSP5) ^ 2 / ((64 + AHPgain * XH(7)) ^ 2 + (PSP5) ^ 2)); %Your Equation Here
        K(6, rk) = DT / TauC * (-XH(6) + 100 * (PSP6) ^ 2 / ((64 + AHPgain * XH(8)) ^ 2 + (PSP6) ^ 2)); %Your Equation Here
        K(7, rk) = DT / TauA * (-XH(7) + XH(5));
        K(8, rk) = DT / TauA * (-XH(8) + XH(6));
        K(9, rk) = DT / TauHT * (-XH(9) + Stim);

    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6; %Most efficient with weight matrix
end;

Calculation_Time = etime(clock, T1)
whitebg('w');
Xiso = 0:100; %X for Isoclines
figure(1); Za = plot(Time, X(1, :), 'r', Time, X(2, :), 'b'); set(Za, 'LineWidth', 2);
xlabel('Time (ms)'); ylabel('Spike Rate');
Eaxis = 0.01:0.5:99;
H = (1 / AHPgain) * (sqrt(- (EE * Eaxis + Stim) .^ 2 + (100 * (EE * Eaxis + Stim) .^ 2) ./ Eaxis) - 64);
figure(2); ZC = plot(X(1, :), X(3, :), '-r', Eaxis, H, '-k', Eaxis, Eaxis, '-k'); set(ZC, 'LineWidth', 2); axis('square');
axis([-5 100 0 60]); title('E-H Projection of State Space');
xlabel('EL'); ylabel('HE');
%Next lines calculate burst rate
Spikes = (X(1, 1:Last - 1) < 30) .* (X(1, 2:Last) >= 30);
SpkTime = zeros(1, sum(Spikes));
Nspk = 1; %Number of spike

for T = 1:length(Spikes); %Calculate spike rate for all interspike intervals
    if Spikes(T) == 1; SpkTime(Nspk) = T * DT; Nspk = Nspk + 1; end;
end;

Final = length(SpkTime);
Rates = 1000 ./ (SpkTime(2:Final) - SpkTime(1:Final - 1));
Rates'
%final lines simulate spikes
BurstL = Make_Spikes(X(1, :), DT, 1);
BurstR = Make_Spikes(X(2, :), DT, 1);
figure(3), plot(Time, BurstL, 'r-', Time, BurstR, 'b-');
xlabel('Time (ms)'); ylabel('V'); title('Approximation of Spikes in each Burst');
