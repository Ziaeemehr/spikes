function Result = NetConv(fltr, inputs)
    % Convolves fltr with inputs and removes extraneous values
    %fltr is assumed to have an odd number of elements and be centered
    %Replicates inputs for periodic boundary conditions
    x = conv(fltr, [inputs inputs inputs]);
    extra = fix(length(fltr) / 2);
    x = x(1 + extra:length(x) - extra);
    Result = x(length(inputs) + 1:2 * length(inputs));
