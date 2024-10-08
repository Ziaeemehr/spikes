%Complete analysis of second order systems & initial value problem
%If both roots are equal (Critical Damping), only the genreral form of the solution is printed.
format loose;
whitebg('w');
clc;
clear all; close all;
CS = -4;
A11 = input('A1 = ');
A12 = input('A2 = ');
A21 = input('A3 = ');
A22 = input('A4 = ');
A = [A11 A12; A21 A22];
B1 = input('B1 = ');
B2 = input('B2 = ');
Init_Condition = [B1 B2]';
Cf = poly(A);
Characteristic_Eqn = Cf
B = Cf(2);
AA = Cf(1);
C = Cf(3);
[EVect, EV] = eig(A);
EigenValue(1) = EV(1, 1);
EigenValue1 = EigenValue(1)
EigenValue(2) = EV(2, 2);
EigenValue2 = EigenValue(2)
EigenVector = EVect;
EV = EigenVector;
EigenVectors = EigenVector
CC = inv(EigenVector) * Init_Condition;
%******************  Code from here on writes out symbolic solutions and does plotting.
if EigenValue1 == EigenValue2;
    EP = num2str(EigenValue1);
    BZ = num2str(B2 - EigenValue1 * B1);
    GenForm = ['x(t) = ', num2str(B1), '*exp(', EP, '*t) + ', BZ, '*t*exp(', EP, '*t)'];
    H2 = figure('position', [5 360 635 200]); clf; Tle = text(0.05, 0.8, 'Special case of Critically Damped Equations', 'fontname', 'palatino', 'fontsize', 18);
    Tx = text(0.1, 0.6, GenForm, 'fontname', 'palatino', 'fontsize', 14, 'color', 'r'); axis off;
    Tf = text(0.15, 0.15, ['Initial Conditions:       x(0)  =  ', num2str(B1), ';           dx(0)/dt  =  ', num2str(B2)], 'fontname', 'palatino', 'fontsize', 14);
end;

if EigenValue1 ~= EigenValue2; %check to exclude Critical Damping
    SC = 'F'; %Flags for writing out correct equations
    PX = 'F';

    if imag(EigenValue(1)) ~= 0;
        SC = 'T';
        if abs(real(EigenValue(1))) > 10 ^ (-6); PX = 'T'; end;
    end;

    StrE1 = [num2str(CC(1) * EV(1, 1)), ' * exp(', num2str(EigenValue(1)), '*t)'];
    StrE2 = [num2str(CC(1) * EV(2, 1)), ' * exp(', num2str(EigenValue(1)), '*t)'];

    if EigenValue(1) == -1; %Eliminate multiplication by �1
        StrE1 = [num2str(CC(1) * EV(1, 1)), ' * exp(-t)'];
        StrE2 = [num2str(CC(1) * EV(2, 1)), ' * exp(-t)'];
    end;

    if EigenValue(1) == 1; %Eliminate multiplication by �1
        StrE1 = [num2str(CC(1) * EV(1, 1)), ' * exp(t)'];
        StrE2 = [num2str(CC(1) * EV(2, 1)), ' * exp(t)'];
    end;

    if SC == 'T'; %sines & cosines
        Cstring = [' * cos(', num2str(abs(imag(EigenValue(1)))), '*t)'];

        if PX == 'T';
            NX = real(EigenValue(1));
            if NX == 1; Cstring = [' * exp(t)', Cstring]; end;
            if NX == -1; Cstring = [' * exp(-t)', Cstring]; end;
            if abs(NX) ~= 1; Cstring = [' * exp(', num2str(NX), '*t)', Cstring]; end;
        end;

        CCa = num2str(2 * real(CC(1) * EV(1, 1)));
        StrE1 = [CCa, Cstring];
        CCb = num2str(2 * real(CC(1) * EV(2, 1)));
        StrE2 = [CCb, Cstring];
        if abs(str2num(CCb)) < 10 ^ (-6); StrE2 = ['']; end;
        if abs(str2num(CCa)) < 10 ^ (-6); StrE1 = ['']; end;
    end;

    if CC(1) * EV(1, 1) == 0;
        StrE1 = [];
    end;

    if CC(1) * EV(2, 1) == 0;
        StrE2 = [];
    end;

    if CC(2) * EV(1, 2) > 0; Pl = ' + '; else Pl = ['  ']; end;
    StrE3 = [Pl, num2str(CC(2) * EV(1, 2)), ' * exp(', num2str(EigenValue(2)), '*t)'];
    if CC(2) * EV(2, 2) > 0; Pl = ' + '; else Pl = ['  ']; end;
    StrE4 = [Pl, num2str(CC(2) * EV(2, 2)), ' * exp(', num2str(EigenValue(2)), '*t)'];

    if SC == 'T'; %sines & cosines
        Cstring = [' * sin(', num2str(abs(imag(EigenValue(2)))), '*t)'];

        if PX == 'T';
            NX = real(EigenValue(1));
            if NX == 1; Cstring = [' * exp(t)', Cstring]; end;
            if NX == -1; Cstring = [' * exp(-t)', Cstring]; end;
            if abs(NX) ~= 1; Cstring = [' * exp(', num2str(NX), '*t)', Cstring]; end;
        end;

        CCa = num2str(2 * imag(CC(2) * EV(1, 2)));
        if str2num(CCa) > 0; Pl = ' + '; else Pl = ['  ']; end;
        StrE3 = [Pl, CCa, Cstring];
        CCb = num2str(2 * imag(CC(2) * EV(2, 2)));
        if str2num(CCb) > 0; Pl = ' + '; else Pl = ['  ']; end;
        StrE4 = [Pl, CCb, Cstring];
        if abs(str2num(CCb)) < 10 ^ (-6); StrE4 = ['']; end;
        if abs(str2num(CCa)) < 10 ^ (-6); StrE3 = ['']; end;
    end;

    if EigenValue(2) == -1; %Eliminate multiplication by �1
        if CC(2) * EV(1, 2) > 0; Sgn = ' + '; else Sgn = ' '; end;
        StrE3 = [Sgn, num2str(CC(2) * EV(1, 2)), ' * exp(-t)'];
        if CC(2) * EV(2, 2) > 0; Sgn = ' + '; else Sgn = ' '; end;
        StrE4 = [Sgn, num2str(CC(2) * EV(2, 2)), ' * exp(-t)'];
    end;

    if EigenValue(2) == 1; %Eliminate multiplication by �1
        StrE3 = [num2str(CC(2) * EV(1, 2)), ' * exp(t)'];
        StrE4 = [num2str(CC(2) * EV(2, 2)), ' * exp(t)'];
    end;

    if CC(2) * EV(1, 2) == 0;
        StrE3 = [];
    end;

    if CC(2) * EV(2, 2) == 0;
        StrE4 = [];
    end;

    X = [StrE1, StrE3];
    Y = [StrE2, StrE4];
    X = ['x(t)  =  ', X];
    Y = ['y(t)  =  ', Y];
    PlotEnd = min(abs(EigenValue)); %For plot scaling
    Tm = [0:0.15 / PlotEnd:15 / PlotEnd];
    Xt = CC(1) * EV(1, 1) * exp(EigenValue(1) * Tm) + CC(2) * EV(1, 2) * exp(EigenValue(2) * Tm);
    Yt = CC(1) * EV(2, 1) * exp(EigenValue(1) * Tm) + CC(2) * EV(2, 2) * exp(EigenValue(2) * Tm);
    Xt = real(Xt);
    Yt = real(Yt);
    H1 = figure('position', [5 20 635 300]); subplot(1, 2, 1), TP = plot(Tm, Xt, 'r-', Tm, Yt, 'b-'); xlabel('Time'); ylabel('Response'); set(TP, 'LineWidth', 2); title('Temporal Response');
    subplot(1, 2, 2), Phz = plot(Xt, Yt, 'r-', 0, 0, 'k+', Xt(1), Yt(1), 'bo'); axis square; xlabel('X'); ylabel('Y');
    title('State Space'); set(Phz, 'LineWidth', 2);
    Minx = min(Xt); Maxx = max(Xt); Miny = min(Yt); Maxy = max(Yt); Xstep = (Maxx - Minx) / 20;; Ystep = (Maxy - Miny) / 20;
    axis([Minx, Maxx, Miny, Maxy]);
    [XX, YY] = meshgrid(Minx:Xstep:Maxx, Miny:Ystep:Maxy);
    DX = A11 * XX + A12 * YY;
    DY = A21 * XX + A22 * YY;
    Scale = 1 ./ sqrt((DX .^ 2 + DY .^ 2));
    DX = DX .* Scale;
    DY = DY .* Scale;
    hold on;
    quiver(XX, YY, DX, DY);
    hold off;
    H2 = figure('position', [5 360 635 200]); clf; Tle = text(0.05, 0.8, 'Analytic Solutions to Second Order Differential Equations', 'fontname', 'palatino', 'fontsize', 18);
    Tx = text(0.1, 0.6, X, 'fontname', 'palatino', 'fontsize', 14, 'color', 'r'); Ty = text(0.1, 0.4, Y, 'fontname', 'palatino', 'fontsize', 14, 'color', 'b'); axis off;
    Tf = text(0.15, 0.15, ['Initial Conditions:       x(0)  =  ', num2str(Init_Condition(1)), ';           y(0)  =  ', num2str(Init_Condition(2))], 'fontname', 'palatino', 'fontsize', 14);
end; %End of IF to avoid critical damping
