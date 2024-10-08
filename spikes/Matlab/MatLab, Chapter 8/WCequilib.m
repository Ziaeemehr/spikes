function ISS = WCequilib(X);
    %Numerically solve for HRW Oscillator Equilibria
    H1 = 1.6 * X + 20 - 30 * sqrt(X / (100 - X));
    H2 = 100 * 2.25 * X ^ 2 / ((900 + 2.25 * X ^ 2));
    ISS = H2 - H1;
