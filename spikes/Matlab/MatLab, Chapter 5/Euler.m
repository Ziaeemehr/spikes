%Simple Euler's Method of DE solution
clear all;
whitebg('w');
H = 0.1; %This is the time step h.
Last = round(10 / H);
X = zeros(1, Last);
X(1) = 2; %initial condition
Y = zeros(1, Last);
Time = H:H:H * Last;

for T = 2:Last;
    X(T) = X(T - 1) + H * (2 * Y(T - 1));
    Y(T) = Y(T - 1) + H * (-2 * X(T - 1));
end;

figure(1), ZZ = plot(Time, X, 'r-', Time, Y, 'b-');
xlabel('Time'); ylabel('X(t) (red) & Y(t) (blue)');
set(ZZ, 'LineWidth', 2);
%Next line shows how to plot phase plane response for two variables
figure(2), FP = plot(X, Y, 'r-'); set(FP, 'LineWidth', 2);
xlabel('X(t)'); ylabel('Y(t)');
