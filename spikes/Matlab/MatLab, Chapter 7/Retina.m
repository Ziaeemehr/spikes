%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all; hold off; clc; close all;
format compact;
Total_Equations = 5; %Solve for this number of interacting Neurons
DT = 2; %Time increment as fraction of time constant
Final_Time = 1000; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
WTS = [1 2 2 1]; %Runge-Kutta Coefficient weights

for NU = 1:Total_Equations; %Initialize
    X(NU, :) = zeros(1, Last); %Vector to store response of Neuron #1
    K(NU, :) = zeros(1, 4); %Runge-Kutta terms
    Weights(NU, :) = WTS; %Make into matrix for efficiency in main loop
end;

Wt2 = [0 .5 .5 1]; %Second set of RK weights
rkIndex = [1 1 2 3];

L = input('Light intensity (0 < L <= 10^4): ');
DelL = input('Test flash increment or decrment (Delta): ');
Rts = roots([9 91 10 -10 * L]);

X(3, 1) = Rts(3); %Bipolar and other response levels in steady state
X(4, 1) = X(3, 1);
G = X(4, 1) / 10;
X(1, 1) = X(3, 1) * (1 + 9 * X(3, 1));
X(2, 1) = X(1, 1);
X(5, 1) = 50 * X(3, 1) / (13 + X(3, 1));

T1 = clock;

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade

        LL = L + DelL * (Tme > 100);
        K(1, rk) = (DT / 10) * (-XH(1) + LL - G * XH(2)); %Your Equation Here
        K(2, rk) = (DT / 100) * (-XH(2) + XH(1)); %Your Equation Here
        K(3, rk) = (DT / 10) * (-XH(3) + (6 * XH(1) - 5 * XH(2)) * (6 * XH(1) > 5 * XH(2)) / (1 + 9 * XH(4)));
        K(4, rk) = (DT / 80) * (-XH(4) + XH(3));
        K(5, rk) = (DT / 10) * (-XH(5) + 50 * XH(3) / (13 + XH(3)));

    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6; %Most efficient with weight matrix
end;

Calculation_Time = etime(clock, T1)
whitebg('w');
figure(1); Za = plot(Time, X(5, :), '-r'); set(Za, 'LineWidth', 2)
xlabel('Time (ms)'); ylabel('Ganglion Cell Spike Rate (Hz)');
Increment = max(X(5, :)) - X(5, 2)
Steady_State = X(5, Last)
