%Ermentrout-Cowan Hallucinations
clc; clear all; close all;
Size = 64; %Spatial size of array
EE = zeros(Size, Size);
IN = zeros(Size, Size);
Stim = zeros(Size, Size);
QStim = zeros(Size, Size);
DX = 20; %microns
Last = 150; %last time in computation
EC = zeros(1, Last);
IC = zeros(1, Last);
X = DX * (1:Size);
DT = 10;
DelT = 1;
[Xsyn, Ysyn] = meshgrid(-Size / 2:(Size / 2 - 1));
Radius = DX * sqrt(Xsyn .^ 2 + Ysyn .^ 2);
Mask = (abs(Xsyn) <= 30) .* (abs(Ysyn) <= 30); %for boundary conditions

%**********
EEgain = 1.95/40;
EIgain = 1.4/60;
IIgain = 2.2/30;
StimTime = 5;
Q = 2.0;
Bias = input('Please choose initial bias (1) Vertical, (2) Horizontal, (3) Oblique, (4) Checks: ');
if Bias == 1; Stim = 3 * cos(2 * pi * Xsyn * 3 / Size); end;
if Bias == 2; Stim = 3 * cos(2 * pi * Ysyn * 3 / Size); end;
if Bias == 3; Stim = 3 * cos(2 * pi * (Xsyn + Ysyn) * 2 / Size); end;
if Bias == 4; Stim = 3 * cos(2 * pi * Xsyn * 2 / Size) .* cos(2 * pi * Ysyn * 2 / Size); end;
Stim = Stim + 3 * (rand(Size, Size) > 0.8);
%**********

synEE = EEgain * exp(-abs(Radius) / 40);
synEE = fft2(synEE);
synEI = EIgain * exp(-abs(Radius) / 60);
synEI = fft2(synEI);
synII = IIgain * exp(-abs(Radius) / 30);
synII = fft2(synII);
whitebg('w');

for T = 1:DelT:Last; %Loop in ms, Euler solution method
    P = Stim * (T <= StimTime) + 1.8;
    QS = QStim * (T <= StimTime) + Q;
    Efft = fft2(EE);
    Ifft = fft2(IN);
    EEresp = real(ifft2(synEE .* Efft - synEI .* Ifft));
    EEresp = fftshift(EEresp) + P;
    EEresp = (EEresp .* (EEresp > 0)) .^ 2;
    INresp = real(ifft2(synEI .* Efft - synII .* Ifft));
    INresp = fftshift(INresp) + QS;
    INresp = (INresp .* (INresp > 0)) .^ 2;
    EE = EE + (DelT / DT) * (-EE + 100 * EEresp ./ (20 ^ 2 + EEresp));
    IN = IN + (DelT / DT) * (-IN + 100 * INresp ./ (40 ^ 2 + INresp));
    EE = 0.5 * EE + 0.5 * EE .* Mask;
    EC(T) = EE(Size / 2);
    IC(T) = IN(Size / 2);

    if rem(T, 2) == 0;
        figure(1); PS = image(EE); colormap(hot(64)); axis square;
        pause(0.2);
    end;

end;
