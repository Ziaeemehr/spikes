function Result = MM(V);
    %Na+ activation variable for Hodgkin-Huxley Equations
    U = -V - 70; %For new Rinzel formulation
    Alpha = 0.1 * (U + 25) ./ (exp((U + 25) / 10) - 1);
    Beta = 4 * exp(U / 18);
    Result = Alpha ./ (Alpha + Beta);
