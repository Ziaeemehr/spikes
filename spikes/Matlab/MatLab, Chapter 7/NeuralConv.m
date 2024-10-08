function Result = NeuralConv(fltr, inputs)
    % Convolves fltr with inputs and removes extraneous values
    %fltr is assumed to have an odd number of elements and be centered
    %Replicates inputs for periodic boundary conditions
    Sz = length(inputs);
    Xx = conv(fltr, inputs);
    extra = fix(length(fltr) / 2);
    Xx = Xx(1 + extra:length(Xx) - extra);
    Result = Xx;
