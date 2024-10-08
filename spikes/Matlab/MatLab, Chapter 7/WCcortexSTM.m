%Wilson-Cowan Equations
whitebg('w');
Size = 100; %Spatial size of array
EE = zeros(1, Size);
IN = zeros(1, Size);
DX = 20; %microns
X = DX * (1:Size);
DT = 10;
DelT = 0.5;
Xsyn = DX * (-15:15);
Stim = zeros(1, Size);
Width = input('Width of stimulus in microns (99 < Width < 1600) = ');
Width = round(Width / (2 * DX));
Stim(Size / 2 - Width:Size / 2 + Width) = ones(1, 2 * Width + 1);
Last = 80; %last time in computation
EC = zeros(1, Last);
IC = zeros(1, Last);

%**********
EEgain = 1.95;
EIgain = 1.4;
IIgain = 2.2;
StimTime = 10;
Q = 0;
%**********

synEE = EEgain * exp(-abs(Xsyn) / 40);
synEI = EIgain * exp(-abs(Xsyn) / 60);
synII = IIgain * exp(-abs(Xsyn) / 30);

for T = 1:DelT:Last; %Loop in ms, Euler solution method
    P = Stim * (T <= StimTime);
    EEresp = NeuralConv(synEE, EE) - NeuralConv(synEI, IN) + P;
    EEresp = (EEresp .* (EEresp > 0)) .^ 2;
    INresp = NeuralConv(synEI, EE) - NeuralConv(synII, IN) + Q;
    INresp = (INresp .* (INresp > 0)) .^ 2;
    EE = EE + (DelT / DT) * (-EE + 100 * EEresp ./ (20 ^ 2 + EEresp));
    IN = IN + (DelT / DT) * (-IN + 100 * INresp ./ (40 ^ 2 + INresp));
    EC(round(T)) = EE(round(Size / 2));
    IC(round(T)) = IN(round(Size / 2));

    if rem(T, 4) == 0;
        figure(1); PS = plot(X, EE, 'r-', X, IN, 'b-'); set(PS, 'LineWidth', 2);
        axis([0 DX * Size 0 110]);
        xlabel('Distance in microns'); ylabel('E (red) & I (blue) Responses');
        pause(0.2);
    end;

end;

Time = 1:Last;
ST = (Time <= StimTime);
figure(2); PT = plot(Time, EC, '-r', Time, IC, 'b-', Time, ST, 'k-'); set(PT, 'LineWidth', 2);
xlabel('Time (ms)'); ylabel('E (red) & I (blue) Responses');
