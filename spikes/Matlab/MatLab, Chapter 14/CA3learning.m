%Simple long-term autoassociative memory network
Stim = zeros(16, 16);
CM = [0 0 0; 1 0 0];
PX = zeros(1, 32);
PY = zeros(1, 32);

for Points = 1:32;
    figure(1), image(Stim + 1), axis square; colormap(CM);
    [PX(Points), PY(Points)] = ginput(1);
    Stim(round(PY(Points)), round(PX(Points))) = 1;
end;

NewImage = Stim;
save NewImage.mat NewImage; %Save created image

Input = 2 * NewImage - 1; %Inhibit cells not driven by pattern
CM = [.5 .5 .5; 1 1 1];
SD = clock; % Three lines to set new random # seed
SD = round((SD(4) + SD(5) + SD(6)) * 10 ^ 3);
rand('seed', SD);
Neurons = zeros(16, 16);
load('Synapses.mat'); %Retrieve modified Synapses file from disk
PSP = zeros(size(Input)); %postsynaptic potential
DT = 10; %time constant in ms
figure(1), image(Input + 1); colormap(CM); axis square;
G = 0; %Inhibitory feedback cell
%Stimulus only on for 20 ms
for T = 1:2:100; %Euler's approximation to solve 256 DE's
    G = G + 2 / DT * (-G + 0.076 * sum(sum(Neurons)));
    PSP(:) = (30 * Input(:) * (T <= 100) + 0.016 * (Neurons(:)' * Synapses)' - 0.1 * G);
    PSP = PSP .* (PSP > 0); %No response for negative inputs
    Neurons(:) = Neurons(:) + 2 / DT * (-Neurons(:) + 100 * (PSP(:) .^ 2) ./ (100 + PSP(:) .^ 2));

    if rem(T - 1, 4) == 0;
        figure(2), image(Neurons); colormap(CM); axis square; pause(0.1);
    end;

end;

NewSynapses = (Neurons(:) > 50) * (Neurons(:) > 50)';

for KK = 1:256; %No self-synapses
    NewSynapses(KK, KK) = 0;
end;

NewSynapses = max(NewSynapses, Synapses);
save NewSynapses.mat NewSynapses;
Input_Resp = max(max(Neurons))

%Recall of all 5 patterns
clear all;
Pat = 1;

while Pat > 0;
    Pat = input('Pattern to recall: (0) Quit, (1) FaceA, (2) FaceB, (3) FW, (4) Assoc, (5) NewImage : ');
    if Pat == 1; load('FaceA.mat'); Stim = FaceA; end;
    if Pat == 2; load('FaceB.mat'); Stim = FaceB; end;
    if Pat == 3; load('FW.mat'); Stim = FW; end;
    if Pat == 4; load('Assoc.mat'); Stim = Assoc; end;
    if Pat == 5; load('NewImage.mat'); Stim = NewImage; end;
    Input = zeros(16, 16);
    CM = [.5 .5 .5; 1 1 1];
    SD = clock; % Three lines to set new random # seed
    SD = round((SD(4) + SD(5) + SD(6)) * 10 ^ 3);
    rand('seed', SD);
    Input(1:9, 1:9) = Stim(1:9, 1:9);
    Noise = input('Amount of Random Noise to add (0 - 100): ');

    for KK = 1:Noise; %add noise to stimulus
        A = floor(14 * rand) + 1;
        B = floor(14 * rand) + 1;
        Input(A, B) = 1;
    end;

    figure(1), image(Input + 1); colormap(CM); axis square; pause(0.1);
    Neurons = zeros(16, 16);
    load('NewSynapses.mat'); %Retrieve modified Synapses file from disk
    Synapses = NewSynapses;
    PSP = zeros(size(Input)); %postsynaptic potential
    DT = 10; %time constant in ms
    figure(1), image(Input + 1); colormap(CM); axis square;
    G = 0; %Inhibitory feedback cell
    %Stimulus only on for 20 ms
    for T = 1:2:120; %Euler's approximation to solve 256 DE's
        G = G + 2 / DT * (-G + 0.076 * sum(sum(Neurons)));
        PSP(:) = (10 * Input(:) * (T <= 20) + 0.016 * (Neurons(:)' * Synapses)' - 0.1 * G);
        PSP = PSP .* (PSP > 0); %No response for negative inputs
        Neurons(:) = Neurons(:) + 2 / DT * (-Neurons(:) + 100 * (PSP(:) .^ 2) ./ (100 + PSP(:) .^ 2));

        if rem(T - 1, 4) == 0;
            figure(2), image(Neurons); colormap(CM); axis square; pause(0.1);
        end;

    end;

    Spike_Rate = max(max(Neurons))
end; %while
