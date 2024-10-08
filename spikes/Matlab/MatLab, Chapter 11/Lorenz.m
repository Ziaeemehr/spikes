%Fourth Order Runge-Kutta for N-Dimensional Systems
clear all;
Total_Equations = 3; %Solve for this number of interacting Neurons
DT = 0.01; %Time increment as fraction of time constant
Final_Time = 50; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
WTS = [1 2 2 1]; %Runge-Kutta Coefficient weights

for NU = 1:Total_Equations; %Initialize
    X(NU, :) = zeros(1, Last); %Vector to store response of Neuron #1
    K(NU, :) = zeros(1, 4); %Runge-Kutta terms
    X(1, 1) = 10; %Initial conditions here if different from zero
    X(2, 1) = 10; %Initial conditions here if different from zero
    X(3, 1) = 40;
    Weights(NU, :) = WTS; %Make into matrix for efficiency in main loop
end;

Wt2 = [0 .5 .5 1]; %Second set of RK weights
rkIndex = [1 1 2 3];
Input = 1.0;
T1 = clock;

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = X(:, T - 1) + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade

        K(1, rk) = DT * 10 * (-XH(1) + XH(2)); %Your Equation Here
        K(2, rk) = DT * (-XH(2) + 28 * XH(1) - XH(1) * XH(3)); %Your Equation Here
        K(3, rk) = DT * (- (8/3) * XH(3) + XH(1) * XH(2));

    end;

    X(:, T) = X(:, T - 1) + sum((Weights .* K)')' / 6; %Most efficient with weight matrix
end;

Calculation_Time = etime(clock, T1)
whitebg('w');
figure(1); Za = plot(Time, X(1, :), 'r'); set(Za, 'LineWidth', 2)
figure(2); Zb = plot(X(1, :), X(2, :), '-r', 0, 0, 'bx', 8.49, 8.48, 'bx', -8.49, -8.49, 'bx'); set(Zb, 'LineWidth', 2);
xlabel('X'); ylabel('Y');
figure(3); Zb = plot(X(1, :), X(3, :), '-r', -8.48, 27, 'bx', 8.48, 27, 'bx'); set(Zb, 'LineWidth', 1);
xlabel('X'); ylabel('Z');
