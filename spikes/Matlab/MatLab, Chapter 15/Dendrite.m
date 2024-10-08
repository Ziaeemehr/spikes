%Postsynaptic potential propagation in a passive dendrite
whitebg('w');
clear all;
X = 0:200; %Length in microns
V1 = 100;
D = 100; %length constant
Tau = 10; %time constant

%**********
NumPSP = 1; %Change this for problems
%**********

X1 = zeros(1, NumPSP);
T1 = zeros(1, NumPSP);

for NN = 1:NumPSP;
    X1(NN) = input('Distance of synapse from soma (10, 200) = ');
    if NN > 1; T1(NN) = input('Time of PSP in ms (integer) = '); end;
end;

PSP = 0;

for T = 0.00001:0.25:10;
    PSP = V1 * exp(-T / Tau) / (2 * D * sqrt(pi * T / Tau)) .* exp(-Tau * (X - X1(1)) .^ 2 / (4 * D ^ 2 * T));

    if NumPSP > 1;

        for S = 2:NumPSP;
            PSP = PSP + (T > T1(S)) * V1 * (exp(- (T - T1(S)) / Tau) / (2 * D * sqrt(pi * (T - T1(S)) / Tau))) .* exp(-Tau * (X - X1(S)) .^ 2 / (4 * D ^ 2 * (T - T1(S))));
        end;

    end;

    figure(1), FF = plot(X, PSP); axis([0, 200, 0, V1 / 50]); set(FF, 'LineWidth', 2); xlabel('Distance in microns');
end;

TT = 0.00001:0.05:10;
PSPtme = zeros(1, length(TT));
PSPtme = V1 * exp(-TT / Tau) ./ (2 * D * sqrt(pi * TT / Tau)) .* exp(-Tau * X1(1) ^ 2 ./ (4 * D ^ 2 * TT));

if NumPSP > 1;

    for S = 2:NumPSP;
        PSPtme = PSPtme + V1 * (TT > T1(S)) .* (exp(- (TT - T1(S)) / Tau) ./ (2 * D * sqrt(pi * (TT - T1(S)) / Tau))) .* exp(-Tau * X1(S) ^ 2 ./ (4 * D ^ 2 * (TT - T1(S))));
    end;

end;

figure(2), F2 = plot(TT, PSPtme, 'r-'); xlabel('Time in ms'); ylabel('Potential Change at Soma');
set(F2, 'LineWidth', 2);
PSPmax = max(PSPtme)
MeanPSP = mean(PSPtme)

disp(PSPtme)
