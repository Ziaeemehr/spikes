function Response = LGNoscillator(XH, Stim, Xinhib);
    %Function for interacting E & I LGN neurons
    K = zeros(1, 14);
    global DT Tau TauR TauC TauH ES IS II TauSyn Tme SynThresh;
    K(1) = DT / Tau * (- (17.81 + 47.58 * XH(1) + 33.8 * XH(1) ^ 2) * (XH(1) - 0.48) - 26 * XH(2) * (XH(1) + 0.95) - 1.93 * XH(3) * (1 - 0.5 * XH(4)) * (XH(1) - 1.4) - 3.25 * XH(4) * (XH(1) + 0.95) - IS * XH(10) * (XH(1) + 0.92));
    K(2) = DT / TauR * (-XH(2) + 1.29 * XH(1) + 0.79 + 3.3 * (XH(1) + 0.38) ^ 2);
    K(3) = DT / TauC * (-XH(3) + 6.65 * (XH(1) + 0.86) * (XH(1) + 0.84));
    K(4) = DT / TauH * (-XH(4) + 3.0 * XH(3));
    K(9) = DT / TauSyn * (-XH(9) + (XH(5) > SynThresh) + (Xinhib > SynThresh));
    K(10) = DT / TauSyn * (-XH(10) + XH(9));

    K(5) = DT / Tau * (- (17.81 + 47.58 * XH(5) + 33.8 * XH(5) ^ 2) * (XH(5) - 0.48) - 26 * XH(6) * (XH(5) + 0.95) - 1.93 * XH(7) * (1 - 0.5 * XH(8)) * (XH(5) - 1.4) - 3.25 * XH(8) * (XH(5) + 0.95) - ES * XH(12) * (XH(5)) + Stim * (Tme >= 20) * (Tme <= 70) - II * XH(14) * (XH(1) + 0.92));
    K(6) = DT / TauR * (-XH(6) + 1.29 * XH(5) + 0.79 + 3.3 * (XH(5) + 0.38) ^ 2);
    K(7) = DT / TauC * (-XH(7) + 6.65 * (XH(5) + 0.86) * (XH(5) + 0.84));
    K(8) = DT / TauH * (-XH(8) + 3.0 * XH(7));
    K(11) = DT / TauSyn * (-XH(11) + (XH(1) > SynThresh));
    K(12) = DT / TauSyn * (-XH(12) + XH(11));
    K(13) = DT / TauSyn * (-XH(13) + (Xinhib > SynThresh));
    K(14) = DT / TauSyn * (-XH(14) + XH(13));
    Response = K';
