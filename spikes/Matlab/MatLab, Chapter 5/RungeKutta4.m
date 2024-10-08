%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all; hold off; clc; close all;

%*****Alter the following four lines for your problem
Total_Equations = 2; %Solve this number of simultaneous equations
H = 0.01; %Time increment
Final_Time = 2; %Final time value for calculation
Initial_Conditions = [1 0]; %Initial condition vector must be length of Total_Equations
%*****

Last = Final_Time / H + 1; %Last time step
Time = H * [0:Last - 1]; %Time vector
WTS = [1 2 2 1]; %Runge-Kutta Coefficient weights

for NU = 1:Total_Equations; %Initialize
    X(NU, :) = zeros(1, Last); %Vector to store response of Neuron #1
    K(NU, :) = zeros(1, 4); %Runge-Kutta terms
    Weights(NU, :) = WTS; %Make into matrix for efficiency in main loop
end;

X(:, 1) = Initial_Conditions';
Wt2 = [0 .5 .5 1]; %Second set of RK weights
rkIndex = [1 1 2 3];
Stim = 0;
T1 = clock;

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * H; %Time upgrade

        K(1, rk) = H / 0.1 * (8 * XH(2)); %Your Equation Here
        K(2, rk) = H / 0.4 * (-2 * XH(1)); %Your Equation Here

    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6; %Most efficient with weight matrix
end;

TrueSol = 2 * Time .* exp(-Time / 20);
Calculation_Time = etime(clock, T1)
whitebg('w');
figure(1); Za = plot(Time, X(1, :), 'r', Time, X(2, :), '-k'); set(Za, 'LineWidth', 2)
Approximation = X(1, Last)
figure(2), SSp = plot(X(1, :), X(2, :), 'r-'); set(SSp, 'LineWidth', 2); axis equal
