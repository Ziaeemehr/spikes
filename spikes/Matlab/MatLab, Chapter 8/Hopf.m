%Uses Routh-Hurwitz to find Oscillatory solution for Hopf Bifurcation
%Choose LONG NUMERIC FORMAT under options for adequate precision in results
function Osc = Hopf(G);
    global A;

    %**********
    A = [-1/20 0 0 -1/20; 1 / G -1 / G 0 0; 0 6/50 -1/50 0; 0 0 1 / G -1 / G]; %Type neural connectivity matrix here
    %**********

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
