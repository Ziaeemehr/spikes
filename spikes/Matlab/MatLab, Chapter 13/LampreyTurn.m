%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all; hold off; clc;
Total_Equations = 10; %Solve for this number of interacting Neurons
DT = 2; %Time increment as fraction of time constant
Final_Time = 1500; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
WTS = [1 2 2 1]; %Runge-Kutta Coefficient weights

for NU = 1:Total_Equations; %Initialize
    X(NU, :) = zeros(1, Last); %Vector to store response of Neuron #1
    K(NU, :) = zeros(1, 4); %Runge-Kutta terms
    X(1, 1) = 1; %Initial conditions here if different from zero
    Weights(NU, :) = WTS; %Make into matrix for efficiency in main loop
end;

Wt2 = [0 .5 .5 1]; %Second set of RK weights
rkIndex = [1 1 2 3];
%**********
Stim1 = input('Strength of Left stimulus = ');
Stim2 = input('Strength of Right stimulus = ');
Ginhib = 2; %Strength of inhibitory cross-coupling
%**********
Tau = 9; %Neural time constants in msec
TauC = 9; %Neural time constants in msec
TauHT = 12; %Time constant for serotonin neuron to have its effect.
EC = 2; % E to C synaptic strength
EE = 5.8; % E to E synaptic strength
T1 = clock;

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade

        TauA1 = 400 / (1 + (0.2 * XH(9)) ^ 2); %Effects of stimulus on tau via 5-HT
        TauAC1 = 400 / (1 + (0.2 * XH(9)) ^ 2);
        AHPgain1 = 4.7 + (0.1 * XH(9)) ^ 2;
        TauA2 = 400 / (1 + (0.2 * XH(10)) ^ 2); %Effects of stimulus on tau via 5-HT
        TauAC2 = 400 / (1 + (0.2 * XH(10)) ^ 2);
        AHPgain2 = 4.7 + (0.1 * XH(10)) ^ 2;
        PSP1 = XH(9) + EE * XH(1) - Ginhib * XH(6);
        PSP1 = PSP1 * (PSP1 > 0);
        PSP2 = XH(10) + EE * XH(2) - Ginhib * XH(5);
        PSP2 = PSP2 * (PSP2 > 0);
        K(1, rk) = DT / Tau * (-XH(1) + 100 * (PSP1) ^ 2 / ((64 + XH(3)) ^ 2 + (PSP1) ^ 2)); %Your Equation Here
        K(2, rk) = DT / Tau * (-XH(2) + 100 * (PSP2) ^ 2 / ((64 + XH(4)) ^ 2 + (PSP2) ^ 2)); %Your Equation Here
        K(3, rk) = DT / TauA1 * (-XH(3) + AHPgain1 * XH(1));
        K(4, rk) = DT / TauA2 * (-XH(4) + AHPgain2 * XH(2));

        PSP5 = XH(9) / 2 + EC * XH(1) - Ginhib * XH(6);
        PSP5 = PSP5 * (PSP5 > 0);
        PSP6 = XH(9) / 2 + EC * XH(2) - Ginhib * XH(5);
        PSP6 = PSP6 * (PSP6 > 0);
        K(5, rk) = DT / TauC * (-XH(5) + 100 * (PSP5) ^ 2 / ((64 + XH(7)) ^ 2 + (PSP5) ^ 2)); %Your Equation Here
        K(6, rk) = DT / TauC * (-XH(6) + 100 * (PSP6) ^ 2 / ((64 + XH(8)) ^ 2 + (PSP6) ^ 2)); %Your Equation Here
        K(7, rk) = DT / TauAC1 * (-XH(7) + AHPgain1 * XH(5));
        K(8, rk) = DT / TauAC2 * (-XH(8) + AHPgain2 * XH(6));
        K(9, rk) = DT / TauHT * (-XH(9) + Stim1);
        K(10, rk) = DT / TauHT * (-XH(10) + Stim2);

    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6; %Most efficient with weight matrix
end;

Calculation_Time = etime(clock, T1)
whitebg('w');
Xiso = 0:100; %X for Isoclines
figure(1); Za = plot(Time, X(1, :), 'r', Time, X(2, :), 'b'); set(Za, 'LineWidth', 2)
figure(2); ZC = plot(X(3, :), X(1, :), '-r'); set(ZC, 'LineWidth', 2); axis('square');
xlabel('A1'); ylabel('E1');
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
BurstL = Make_Spikes(X(1, :), DT, 1);
BurstR = Make_Spikes(X(2, :), DT, 1);
figure(3), BP = plot(Time, BurstL, 'r-', Time, BurstR, 'b-');
