%Spike propagation via diffusion using HRW Hodgkin-Huxley Equations
clear all;
whitebg('w');
Tau = 0.8;
TauR = 1.9;
DT = 0.01; %Time increment as fraction of time constant
DX = 0.08; %Delta X spatial increment
X = 0:DX:12;
Right = length(X);
Final_Time = 10; %Final time value for calculation
Last = Final_Time / DT + 1; %Last time step
Time = DT * [0:Last - 1]; %Time vector
V = -0.704 * ones(1, Right); %Resting potential is -0.704
R = 0.088 * ones(1, Right);
Choice = input('Please choose (1) Stimulate left, (2) Stimulate Center, or (3) Stimulate both ends: ');

%**********
ENa = 0.55; %change for problems
DD = 0.25; %Length constant for diffusion
%**********

%Initial conditions
if Choice == 1 | Choice == 3; V(1) = -0.065; end; %Excitation at first location
if Choice == 3; V(Right) = -0.065; end;
if Choice == 2; V(round(Right / 2)) = -0.065; end;
figure(1), ZA = plot(X, 100 * V, 'r-'); set(ZA, 'LineWidth', 3);
axis([0 12 -100 50]);
xlabel('Distance along Axon (mm)'); ylabel('Membrane Voltage (mV)');
title('Spike Propagation during 10 msec');
pause(0.1);
Pos = 1; %position index for velocity

for T = 2:Last;
    VV = V;
    VL = [V(1), V(1:Right - 1)]; %Incorporates zero flux boundary conditions.
    VR = [V(2:Right), V(Right)];
    V = V + DT / Tau * (- (17.81 + 47.71 * VV + 32.63 * VV .^ 2) .* (VV - ENa) - 26.0 * R .* (VV + 0.92) + (DD ^ 2 / DX ^ 2) * (VL + VR - 2 * VV));
    R = R + DT / TauR * (-R + 1.35 * VV + 1.03);

    if rem((T - 2), 40) == 0;
        ZA = plot(X, 100 * V, 'r-'); set(ZA, 'LineWidth', 3); axis([0 12 -100 50]);
        xlabel('Distance along Axon (mm)'); ylabel('Membrane Voltage (mV)');
        title('Spike Propagation during 10 msec');
        pause(0.1);
    end;

    if (T * DT == 3) | (T * DT == 6); %Calculate positions to determine velocity
        [Mx, J(Pos)] = max(V);
        Pos = Pos + 1;
    end;

end;

Spike_Velocity = (X(J(2)) - X(J(1))) / 3
