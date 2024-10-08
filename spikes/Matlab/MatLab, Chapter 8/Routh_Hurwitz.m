%Routh-Hurwitz & Eigenvalues
%Accurate to about 10^(-9).
clc;
clear;
format short;
global A;
%Guess = input('Please specify initial guess: ');
G = 10;
A = [-1/20 0 0 -1/20; 1 / G -1 / G 0 0; 0 6/50 -1/50 0; 0 0 1 / G -1 / G];
G = -3.2470e-16; %G = fzero('Hopf', Guess);
Stimulus = [1 1 1]';
Cf = poly(A);
Characteristic_Eqn = Cf;
EigenValues = eig(A);
EigenValues = 10 ^ (-9) * round(EigenValues * 10 ^ 9);

if sum(Cf >= 0) ~= length(Cf); %Check to make sure that all coefficients are > zero
    'Equilibrium point is unstable because some characteristic eqn coefficients are <= 0!'
end;

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

for K = 1:Sz;
    RHDet(K) = det(RH(1:K, 1:K));
end;

RHDet = RHDet .* (abs(RHDet) > 10 ^ (-9));
RHDeterminants = RHDet';

size(RHDet)

if sum(RHDet > 0) == length(RHDet);
    'Equilibrium point is asymptotically stable!'
end;

if sum(RHDet < -10 ^ (-6)) > 0;
    'Equilibrium Point is Unstable!'
end;

if RHDet(Sz - 1) == 0;
    'Solution oscillates around equilibrium point, which is a center!'
end;
