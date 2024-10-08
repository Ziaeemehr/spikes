function Result = Vnull(W);
    %dV/dt = 0 isocline for Rinzel
    global VX IE;
    S = 1.2714;
    U = -VX - 70; %For new Rinzel formulation
    Alpha = 0.1 * (U + 25) ./ (exp((U + 25) / 10) - 1);
    Beta = 4 * exp(U / 18);
    M = Alpha ./ (Alpha + Beta + 0.000001);
    Result = (IE - 120 * M ^ 3 * (1 - W) * (VX - 55) - 36 * (W / S) ^ 4 * (VX + 92) - 0.3 * (VX + 50.528)) ^ 2;
