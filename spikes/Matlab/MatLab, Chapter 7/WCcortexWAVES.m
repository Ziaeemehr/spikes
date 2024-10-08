%Wilson-Cowan Equations
whitebg('w');
Size = 400; %Spatial size of array
EE = zeros(1, Size);
IN = zeros(1, Size);
Stim = zeros(1, Size);
Width = 100;
DX = 20; %microns
Width = round(Width / (2 * DX));
Last = 80; %last time in computation
EC = zeros(1, Last);
IC = zeros(1, Last);
X = DX * (1:Size);
DT = 10;
DelT = 0.5;
Xsyn = DX * (-15:15);

%**********
EEgain = 1.9;
EIgain = 1.5;
IIgain = 1.5;
StimTime = 5;
Q = -90;

%To observe wave anniahilation, comment out first line and activate other two below
Stim(Size / 2 - Width:Size / 2 + Width) = 2 * ones(1, 2 * Width + 1);
%Stim(61:70) = 4*ones(1, 10);
%Stim(Size - 69:Size-60) = 4*ones(1, 10);
%**********

synEE = EEgain * exp(-abs(Xsyn) / 40);
synEI = EIgain * exp(-abs(Xsyn) / 60);
synII = IIgain * exp(-abs(Xsyn) / 30);

for T = 1:DelT:Last; %Loop in ms, Euler solution method
    P = Stim * (T <= StimTime);
    EEresp = CircleConv(synEE, EE) - CircleConv(synEI, IN) + P;
    EEresp = (EEresp .* (EEresp > 0)) .^ 2;
    INresp = CircleConv(synEI, EE) - CircleConv(synII, IN) + Q;
    INresp = (INresp .* (INresp > 0)) .^ 2;
    EE = EE + (DelT / DT) * (-EE + 100 * EEresp ./ (20 ^ 2 + EEresp));
    IN = IN + (DelT / DT) * (-IN + 100 * INresp ./ (40 ^ 2 + INresp));
    EC(round(T)) = EE(Size / 2);
    IC(round(T)) = IN(Size / 2);

    if rem(T, 2) == 0;
        figure(1); PS = plot(X, EE, 'r-', X, IN, 'b-'); set(PS, 'LineWidth', 2);
        axis([0 DX * Size 0 100]);
        xlabel('Distance in microns'); ylabel('E (red) & I (blue) Responses');
        pause(0.2);
    end;

end;
