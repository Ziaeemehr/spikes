%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all; hold off;
Total_Neurons = 13; %Solve for this number of interacting Neurons
DT = 0.05; %Time increment as fraction of time constant
Final_Time = 4000; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
Tau = 0.8; %Neural time constants in msec
TauR = 1.9;
TauA = 0.97;
TauAR = 5.6;
TauE = 40; %EPSP time constant

%**********
TauED = 320; %EPSP time constant for D neurons
%**********

TauI = 10; %IPSP time constant
TauH = 1250;
WTS = [1 2 2 1]; %Runge-Kutta Coefficient weights

for NU = 1:Total_Neurons; %Initialize
    X(NU, :) = zeros(1, Last); %Vector to store response of Neuron #1
    K(NU, :) = zeros(1, 4); %Runge-Kutta terms
    Weights(NU, :) = WTS; %Make into matrix for efficiency in main loop
end;

X(1, 1) = -0.70; %Initial conditions here if different from zero
X(2, 1) = 0.088; %Initial conditions here if different from zero
X(5, 1) = -0.70; %Initial conditions here if different from zero
X(6, 1) = 0.088; %Initial conditions here if different from zero
X(8, 1) = -0.754; %Initial conditions here if different from zero
X(9, 1) = 0.279; %Initial conditions here if different from zero

Wt2 = [0 .5 .5 1]; %Second set of RK weights
rkIndex = [1 1 2 3];
Stim = 0.4;
ES = 7;
HP = input('Magnitude of Iahp current = ');
whitebg('w');
T1 = clock;
ST = 10.6;

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade
        Stim1 = Stim * (Tme >= 50) * (Tme <= 100);

        %DSI neuron
        K(1, rk) = DT / Tau * (- (17.81 + 47.71 * XH(1) + 32.63 * XH(1) ^ 2) * (XH(1) - 0.55) - 26 * XH(2) * (XH(1) + 0.92) - HP * XH(12) * (XH(1) + 0.92) + Stim1 - ES * XH(3) * XH(1) - 9 * XH(4) * (XH(1) + 0.92));
        K(2, rk) = DT / TauR * (-XH(2) + 1.35 * XH(1) + 1.03);
        K(3, rk) = DT / TauED * (-XH(3) + 2 * (XH(1) > -0.5));
        K(4, rk) = DT / TauI * (-XH(4) + 2 * (XH(8) > -0.5)); %Inhibition from VSI
        K(12, rk) = DT / TauH * (-XH(12) + 11 * (XH(1) + 0.70) * (XH(1) + 0.65));

        %C2 neuron
        K(5, rk) = DT / Tau * (- (17.81 + 47.71 * XH(5) + 32.63 * XH(5) ^ 2) * (XH(5) - 0.55) - 26 * XH(6) * (XH(5) + 0.92) - 0.5 * XH(13) * (XH(5) + 0.92) - 0.6 * XH(7) * XH(5) - 0.25 * XH(4) * (XH(5) +0.92));
        K(6, rk) = DT / TauR * (-XH(6) + 1.35 * XH(5) + 1.03);
        K(7, rk) = DT / TauE * (-XH(7) + 3 * (XH(1) > -0.5));
        K(13, rk) = DT / TauH * (-XH(13) + 11 * (XH(5) + 0.70) * (XH(5) + 0.65));

        %VSI neurons with IA current
        K(8, rk) = DT / TauA * (- (17.81 + 47.58 * XH(8) + 33.8 * XH(8) ^ 2) * (XH(8) - 0.48) - 26 * XH(9) * (XH(8) + 0.95) - 2.5 * XH(10) * XH(8) - 0.8 * XH(11) * (XH(8) +0.92));
        K(9, rk) = DT / TauAR * (-XH(9) + 1.29 * XH(8) + 0.79 + 3.3 * (XH(8) + 0.38) ^ 2);
        K(10, rk) = DT / TauE * (-XH(10) + 1.5 * (XH(5) > -0.5) + 4.0 * (XH(8) > -0.5));
        K(11, rk) = DT / TauI * (-XH(11) + 3 * (XH(1) > -0.5)); %Inhibition from V1 & V2

        if rem(Tme, 400) == 0;
            figure(1), ZA = plot(Time, 100 * X(1, :), 'r-', Time, 100 * X(8, :) - 150, 'b-', Time, 100 * X(5, :) - 300, 'k-'); set(ZA, 'LineWidth', 1)
            xlabel('Time (ms)');
            ylabel('Potential (lower spike trains shifted down)');
            pause(0.1);
        end;

    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6;
end;

Calculation_Time = etime(clock, T1)
figure(1), ZA = plot(Time, 100 * X(1, :), 'r-', Time, 100 * X(8, :) - 150, 'b-', Time, 100 * X(5, :) - 300, 'k-'); set(ZA, 'LineWidth', 1)
xlabel('Time (ms)');
ylabel('Potential (lower spike trains shifted down)');
