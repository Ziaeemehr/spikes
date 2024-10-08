%Uses Routh-Hurwitz to find Oscillatory solution for Hopf Bifurcation
function Osc = ChatteringHopf(G);
    global A;
    A = [-570.62 * G ^ 2 - 715.53 * G - 228.08, -26.804 * (G + 0.95), -1.7526 * (G - 1.4), -13.402 * (G + 0.95);
         3.1429 * G + 1.8086, -0.47619, 0, 0;
         (6/5) * G + 0.8724, 0, -1/15, 0;
         0, 0, 3/56, -1/56]; %Type your neural matrix here
    Cf = poly(A);
    Sz = length(Cf) - 1;
    RH = zeros(Sz, Sz);

    for Row = 1:Sz; %set-up Routh-Hurwitz matrix
        Index = 2 * Row +1;

        for Column = 1:Sz;
            Index = Index - 1;

            if (Index > 0) & (Index <= (Sz + 1));
                RH(Row, Column) = Cf(Index);
            end;

        end;

    end;

    Osc = det(RH(1:Sz - 1, 1:Sz - 1));
