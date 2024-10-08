%Wilson-Cowan Equations
whitebg('w');
Size = 200; %Spatial size of array
EE = 0.25 * ones(1, Size);
IN = 3.85 * ones(1, Size);
DX = 20; %microns
X = DX * (1:Size);
DT = 10;
DelT = 0.5;
Xsyn = DX * (-15:15);
Stim = ones(1, Size);
Last = 300; %last time in computation
EC = zeros(1, Last);
IC = zeros(1, Last);
freq = input('Spatial Frequency (integer from 0-50) for stimulus = ');

%**********
EEgain = 1.95;
EIgain = 1.4;
IIgain = 2.2;
freq = freq / max(X);
%**********

synEE = EEgain * exp(-abs(Xsyn) / 40);
synEI = EIgain * exp(-abs(Xsyn) / 60);
synII = IIgain * exp(-abs(Xsyn) / 30);

for T = 1:DelT:Last; %Loop in ms, Euler solution method
    P = 31.5 * Stim + 0.01 * cos(2 * pi * freq * X) * (T <= 5);
    Q = 32.3 * Stim;
    EEresp = CircleConv(synEE, EE) - CircleConv(synEI, IN) + P;
    EEresp = (EEresp .* (EEresp > 0)) .^ 2;
    INresp = CircleConv(synEI, EE) - CircleConv(synII, IN) + Q;
    INresp = (INresp .* (INresp > 0)) .^ 2;
    EE = EE + (DelT / DT) * (-EE + 100 * EEresp ./ (20 ^ 2 + EEresp));
    IN = IN + (DelT / DT) * (-IN + 100 * INresp ./ (40 ^ 2 + INresp));
    EC(round(T)) = EE(Size / 2);
    IC(round(T)) = IN(Size / 2);

    if rem(T, 5) == 0;
        figure(1); PS = plot(X, EE, 'r-', X, IN, 'b-'); set(PS, 'LineWidth', 2);
        axis([0 DX * Size 0 110]);
        xlabel('Distance in microns'); ylabel('E (red) & I (blue) Responses');
        pause(0.1);
    end;

end;

Time = 1:Last;
