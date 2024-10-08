%24 Neuron Vector Sum WTA network
clc; clear all; close all;
Neurons = zeros(1, 24);
Directions = (-180:15:165) * pi / 180; %Preferred directions
NV = 2;

%**********
%NV = input('Number of input vectors to network (<= 4) = ');
%**********

Tau = 10;
Size = zeros(1, NV);
Ang = zeros(1, NV);

for K = 1:NV;
    Number = K
    Size(K) = input('Length of Vector (>= 1) = ');
    Ang(K) = input('Vector Angle in deg re. Horizontal = ');
end;

whitebg('w');
Ang = Ang * pi / 180; %convert to radians
Inputs = zeros(1, 24);

for K = 1:NV;
    Inputs = Inputs + Size(K) * cos(Directions - Ang(K)) .* (cos(Directions - Ang(K)) >= 0);
end;

figure(1), IP = polar(Directions, Inputs); set(IP, 'LineWidth', 2);
Resp = zeros(1, 100);
Isynapses = -3 * [1 1 1 1 1 1 0 0 0 0 0 1 1 1 1 1 1]; %Spread of feedback inhibition

for T = 2:1:100;
    PSP = Inputs + CircleConv(Isynapses, Neurons);
    PSP = (PSP) .* (PSP > 0); %Threshold of 0
    Neurons = Neurons + (2 / Tau) * (-Neurons + PSP);
    Resp(T) = Neurons(13);
end;

figure(2), ZB = polar(Directions, Neurons, 'r-'); set(ZB, 'LineWidth', 2);
figure(3), TB = plot(1:100, Resp, 'r-'); set(TB, 'LineWidth', 2);

%These lines compute the perceived output direction with Parabolic interpolation
[MAX, jj] = max(Neurons);
plotorient =- [-180:15:165];
PerceivedDirection = ParabolaInt(Neurons(jj - 1), Neurons(jj), Neurons(jj + 1), plotorient(jj), 15.0)
TrueVectorDirection = atan2(sum(Size .* sin(Ang)), sum(Size .* cos(Ang))) * 180 / pi
DirectionError = TrueVectorDirection - PerceivedDirection
PerceivedLength = MAX
TrueVectorLength = sum(Size .* cos(Ang - TrueVectorDirection * pi / 180))
