%Computation of Lyapunov exponent for Chaos
%No variables are cleared so all X(N) exist from previous calculation

Y = X; %Y now contains the trajectory we will use
DelD = input('Distance Increment for X(1) = ');
WTS = [1 2 2 1]; %Runge-Kutta Coefficient weights

for NU = 1:Total_Equations; %Initialize
    X(NU, :) = zeros(1, Last); %Vector to store response of Neuron #1
    K(NU, :) = zeros(1, 4); %Runge-Kutta terms
    X(:, 1) = Y(:, 1); %Initial conditions
    DZ = zeros(1, Total_Equations);
    DZ(1) = DelD; %initial neighboring trajectory
    DZ = DZ'; %Make into column vector
    Weights(NU, :) = WTS; %Make into matrix for efficiency in main loop
end;

Ratio = zeros(1, Last - 1); %Store divergence distance ratios
Wt2 = [0 .5 .5 1]; %Second set of RK weights
rkIndex = [1 1 2 3];
T1 = clock;

for T = 2:Last;

    for rk = 1:4 %Fourth Order Runge-Kutta
        XH = Y(:, T - 1) + DZ + K(:, rkIndex(rk)) * Wt2(rk);
        Tme = Time(T - 1) + Wt2(rk) * DT; %Time upgrade

        ST = Stim + Amp * sin(2 * pi * Tme / 3.78);
        K(1, rk) = DT / Tau * (- (17.81 + 47.71 * XH(1) + 32.63 * XH(1) ^ 2) * (XH(1) - 0.55) - 26 * XH(2) * (XH(1) + 0.92) + ST);
        K(2, rk) = DT / TauR * (-XH(2) + 1.35 * XH(1) + 1.03);

    end;

    X(:, T) = Y(:, T - 1) + DZ + sum((Weights .* K)')' / 6; %Most efficient with weight matrix
    Diff = Y(:, T) - X(:, T);
    RelDist = sqrt(sum(Diff .^ 2)) / DelD; %Ratio change in distance from trajectory
    Ratio(T - 1) = RelDist;
    DZ = Diff / RelDist;
end;

Calculation_Time = etime(clock, T1)
whitebg('w');
Lyapunov_Exponent = 1 / (DT * length(Ratio)) * sum(log(Ratio))
