%Calls Wilson-Cowan oscillator equilibrium function
Guess = input('Initial guess at solution E = ');
E = fzero('WCequilib', Guess)
I = 100 * (1.5 * E) ^ 2 / (30 ^ 2 + (1.5 * E) ^ 2)
