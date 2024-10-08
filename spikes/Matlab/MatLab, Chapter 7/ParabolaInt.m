function PB = ParabolaInt(Rminus1, Rmax, Rplus1, MaxX, Del);
    %Function performs parabolic interpolation on inputs
    Num = Del * (Rplus1 - Rminus1);
    Denom = 2.0 * (Rplus1 + Rminus1 - 2.0 * Rmax);
    PB = -Num / Denom - MaxX;
