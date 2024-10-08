%Wilson-Cowan Equations
whitebg('w');
Size = 100; %Spatial size of array
EE = zeros(1, Size);
IN = zeros(1, Size);
Stim = zeros(1, Size);
Width = 5;
Width = round(Width / 2);
Stim(Size / 2 - Width:Size / 2 + Width) = 1.4 * ones(1, 2 * Width + 1);
Last = 50; %last time in computation
EC = zeros(1, Last);
IC = zeros(1, Last);
DX = 20; %microns
X = DX * (1:Size);
DT = 10;
DelT = 0.5;
Xsyn = DX * (-15:15);

%**********
EEgain = 1.5;
EIgain = 1.3;
IIgain = 1.5;
StimTime = 5;
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
    EC(round(T)) = EE(Size / 2);
    IC(round(T)) = IN(Size / 2);

    if rem(T, 2) == 0;
        figure(1); PS = plot(X, EE, 'r-', X, IN, 'b-'); set(PS, 'LineWidth', 2);
        axis([0 DX * Size 0 30]);
        xlabel('Distance in microns'); ylabel('E (red) & I (blue) Responses');
        pause(0.2);
    end;

end;

Time = 1:Last;
ST = (Time <= StimTime);
figure(2); PT = plot(Time, EC, '-r', Time, IC, 'b-', Time, ST, 'k-'); set(PT, 'LineWidth', 2);
xlabel('Time (ms)'); ylabel('E & I Responses');
