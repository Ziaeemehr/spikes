%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all; hold off; clc; close all;
Total_Neurons = 2; %Solve for this number of interacting Neurons
DT = 0.2; %Time increment as fraction of time constant
Final_Time = 100; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
Tau = 20; %Neural time constants in msec
WTS = [1 2 2 1]; %Runge-Kutta Coefficient weights

for NU = 1:Total_Neurons; %Initialize
    X(NU, :) = zeros(1, Last); %Vector to store response of Neuron #1
    K(NU, :) = zeros(1, 4); %Runge-Kutta terms
    X(1, 1) = 0; %Initial conditions here if different from zero
    X(2, 1) = 0; %Initial conditions here if different from zero
    Weights(NU, :) = WTS; %Make into matrix for efficiency in main loop
end;

Wt2 = [0 .5 .5 1]; %Second set of RK weights
rkIndex = [1 1 2 3];
Stim = 5000 %input('Light level (0-10000) = ');
T1 = clock;

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade

        K(1, rk) = DT / 10 * (-XH(1) + Stim / (1 + XH(2))); %Your Equation Here
        K(2, rk) = DT / 10 * (-XH(2) + 2 * XH(1)); %Your Equation Here

    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6; %Most efficient with weight matrix
end;

Calculation_Time = etime(clock, T1)
whitebg('w');
Xiso = 0:0.2:10; %X for Isoclines
Isocline2 = Xiso / 2;
Xiso1 = 0:0.2:10;
Isocline1 = Stim ./ (1 + Xiso1);
figure(1); Za = plot(Time, X(1, :), 'r', Time, X(2, :), 'b'); set(Za, 'LineWidth', 2)
xlabel('Time (ms)'); ylabel('B(t) (red) & A(t) (blue)');
figure(2); Zb = plot(X(1, :), X(2, :), '-r', Isocline1, Xiso1, '-k', Isocline2, Xiso, '--k'); set(Zb, 'LineWidth', 2); axis square;
xlabel('B'); ylabel('A'); title('Phase plane, Isoclines (black) & Trajectory (red)');
