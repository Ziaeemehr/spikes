%Applies FFT, First return map & Liapunov exponent to time series
whitebg('w');
T = 0:0.002:40.96;

%***** Choose ONLY ONE appropriate line below
%***** Compare chaos with noise, periodicity & quasiperiodicity
%TS =  2*sin(2*pi*2*T) + 3*sin(2*pi*5*T);
TS = 2 * sin(2 * pi * 2 * T) + 3 * sin(2 * pi * sqrt(23) * T);
%TS = randn(1, 10000);
TS = X(1, :);
%*****

Last = length(TS);
figure(1), PX = plot(TS, 'r-'); set(PX, 'LineWidth', 2);
xlabel('Time'); ylabel('Time Series');
Period = input('Period at which to sample series (integer) = ');
Offset = input('Offest from start to begin sampling (integer) = ');
Y = TS - sum(TS) / Last; %subtract out average value
FFT = fft(Y) * 2 / Last;
Power = (abs(FFT) .^ 2);
MX = max(Power);
figure(2), P2 = semilogy(Power(1:Last / 8), 'r-');
axis([1, Last / 8, MX / 10 ^ 8, MX * 3]); set(P2, 'LineWidth', 1);
xlabel('Frequency'); ylabel('Power'); title('Fourier Power Psectrum');
Returns = zeros(1, Last / 4);
Previous = 0;
NumRet = 0;

for K = Offset:Period:Last;
    NumRet = NumRet + 1;
    Returns(NumRet) = TS(K);
    Previous = K;
end;

NextReturn = Returns(12:NumRet); %Discount first 10 for transients
Returns = Returns(11:NumRet - 1);
figure(3); RM = plot(Returns, NextReturn, 'r.'); set(RM, 'LineWidth', 2);
xlabel('Return(T)'); ylabel('Return(T+1)'); title('Return Map');
Returns = length(NextReturn)
