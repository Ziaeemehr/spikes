function SpkTrain = Make_Spikes(Rate, Dt, Criterion);
    %Program to simulate individual spikes from a spike rate neuron
    %Dt is the interval in ms between entries in the Rate vector
    %Criterion represents sampling rate: 1 = veridical; 2, 3 = every 2nd, 3rd spike,
    SZ = length(Rate);
    Spikes = -70 * ones(1, SZ); %-70 mV resting potential
    MSRate = Rate * Dt / 1000; %Rate per Dt ms
    Last = 1;

    for K = 1:(SZ - 1);
        Spk = sum(MSRate(Last:K));

        if Spk >= Criterion;
            Spikes(K) = 25;
            Spikes(K + 1) = -82;
            Last = K + 1;
        end;

    end;

    SpkTrain = Spikes + Rate / 10; %Add a simulated PSP
