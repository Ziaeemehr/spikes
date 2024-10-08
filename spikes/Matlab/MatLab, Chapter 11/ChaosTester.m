%Applies FFT, First return map & Liapunov exponent to time series
%For large numbers of returns, program takes a long time
whitebg('w');
T = 0:0.002:40.96;

Choice = input('Test (1) oscillation, (2) quasiperiodic, (3) random, (4) Lorenz, (5) Hodgkin-Huxley:');
if Choice == 1; TS = 2 * sin(2 * pi * 2 * T) + 3 * sin(2 * pi * 5 * T); end;
if Choice == 2; TS = 2 * sin(2 * pi * 2 * T) + 3 * sin(2 * pi * sqrt(23) * T); end;
if Choice == 3; TS = randn(1, 100000); end;
if Choice == 4; TS = X(3, :); end;
if Choice == 5; TS = X(1, :); end;
%*****

Last = length(TS);
figure(1), PX = plot(TS, 'r-'); set(PX, 'LineWidth', 2);
xlabel('Time'); ylabel('Time Series');
Value = input('Value for calculating First Return Map = ');
T1 = clock;
Y = TS - sum(TS) / Last; %subtract out average value
FFT = fft(Y) * 2 / Last;
Power = (abs(FFT) .^ 2);
MX = max(Power);
figure(2), P2 = semilogy(Power(1:round(Last / 8)), 'r-');
axis([1, Last / 8, MX / 10 ^ 8, MX * 3]); set(P2, 'LineWidth', 1);
xlabel('Frequency'); ylabel('Power'); title('Fourier Power Spectrum');
Returns = zeros(1, round(Last / 4));
NumRet = 0;
Previous = 0;

for K = 2:Last;

    if (TS(K - 1) <= Value) & (TS(K) >= Value); %Count only positive direction
        NumRet = NumRet + 1;
        Returns(NumRet) = K - Previous;
        Previous = K;
    end;

end;

NextReturn = Returns(12:NumRet); %Discount first 10 for transients
Returns = Returns(11:NumRet - 1);
figure(3); RM = plot(Returns, NextReturn, 'r.'); set(RM, 'LineWidth', 2);
xlabel('Return(T)'); ylabel('Return(T+1)'); title('First Return Map');
Returns = length(NextReturn)
Calculation_Time = etime(clock, T1)
