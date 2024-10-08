function Result = GG(V);
    %Equilibrium value of recovery variable W for Hodgkin-Huxley Equations
    S = 1.2714;
    U = -V - 70; %for new Rinzel system
    AH = 0.07 * exp(U / 20);
    BH = (1 + exp((U + 30) / 10)) .^ (-1);
    Hh = AH ./ (AH + BH); %Na+ inactivation variable
    AN = 0.01 * (U + 10) ./ (exp((U + 10) / 10) - 1);
    BN = 0.125 * exp(U / 80);
    Nn = AN ./ (AN + BN);
    Result = S * (Nn + S * (1 - Hh)) ./ (1 + S ^ 2);
