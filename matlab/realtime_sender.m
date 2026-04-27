% realtime_sender.m
% --------------------------------------------------------------------------
% Streams synthetic IEEE 1159 PQ disturbance signals to the PQD backend
% over TCP (port 5555) as newline-delimited JSON frames.
%
% Each frame: {"label":"<ClassName>","signal":[v0,...,v99],"window":N}\n
%
% In demo mode the script cycles through all 17 classes, holding each
% class for HOLD_WINDOWS frames and then advancing to the next.
% --------------------------------------------------------------------------

clear; clc;

% ── Configuration ────────────────────────────────────────────────────────
HOST          = '127.0.0.1';
PORT          = 5555;
MODE          = 'demo';        % 'demo' is the only mode supported
SEND_PERIOD_S = 0.020;         % 50 windows / sec  (20 ms per frame)
HOLD_WINDOWS  = 50;            % stay on each class for this many frames
SNR_DB        = 40;            % AWGN SNR in dB
RECONNECT_S   = 2;             % seconds to wait before retrying socket

fs   = 5000;
f0   = 50;
N    = 100;
A    = 1.0;
t    = (0:N-1) / fs;

CLASSES = { ...
    'Pure_Sinusoidal','Sag','Swell','Interruption','Transient', ...
    'Oscillatory_Transient','Harmonics','Flicker','Notch', ...
    'Sag_Harmonics','Sag_Flicker','Sag_Oscillatory', ...
    'Swell_Harmonics','Swell_Flicker','Swell_Oscillatory', ...
    'Harmonics_Flicker','Harmonics_Notch'};

fprintf('PQD MATLAB sender starting in %s mode.\n', MODE);
fprintf('Target: %s:%d   period: %.0f ms   classes: %d\n', ...
    HOST, PORT, SEND_PERIOD_S * 1000, numel(CLASSES));

windowIdx = 0;

while true
    % ── Connect (or reconnect) ───────────────────────────────────────────
    client = [];
    try
        client = tcpclient(HOST, PORT, 'Timeout', 5);
        configureTerminator(client, "LF");
        fprintf('Connected to %s:%d\n', HOST, PORT);
    catch ME
        fprintf('Connect failed (%s). Retrying in %ds...\n', ...
            ME.message, RECONNECT_S);
        pause(RECONNECT_S);
        continue
    end

    % ── Streaming loop ───────────────────────────────────────────────────
    try
        while true
            classIdx = mod(floor(windowIdx / HOLD_WINDOWS), numel(CLASSES)) + 1;
            cls = CLASSES{classIdx};

            sig = generate_signal(cls, t, f0, A);
            sig = add_awgn(sig, SNR_DB);

            % Build JSON frame
            frame = struct( ...
                'label',  cls, ...
                'signal', sig, ...
                'window', windowIdx);
            jsonStr = jsonencode(frame);
            payload = uint8([jsonStr newline]);   % LF-terminated

            write(client, payload, 'uint8');

            if mod(windowIdx, 10) == 0
                fprintf('[w=%6d] %s\n', windowIdx, cls);
            end

            windowIdx = windowIdx + 1;
            pause(SEND_PERIOD_S);
        end
    catch ME
        fprintf('Send loop ended (%s). Reconnecting...\n', ME.message);
        try
            clear client
        catch
        end
        pause(RECONNECT_S);
    end
end


% ══════════════════════════════════════════════════════════════════════════
% Local helpers
% ══════════════════════════════════════════════════════════════════════════

function sig = generate_signal(cls, t, f0, A)
% Generate one 100-sample window of the given disturbance class.
    base = A * sin(2*pi*f0*t);
    switch cls
        case 'Pure_Sinusoidal'
            sig = base;

        case 'Sag'
            alpha = rand_range(0.1, 0.9);
            sig = A * (1 - alpha) * sin(2*pi*f0*t);

        case 'Swell'
            alpha = rand_range(0.1, 0.7);
            sig = A * (1 + alpha) * sin(2*pi*f0*t);

        case 'Interruption'
            sig = A * 0.05 * sin(2*pi*f0*t);

        case 'Transient'
            B   = rand_range(0.3, 0.9) * A;
            tau = rand_range(0.0002, 0.0007);
            ts  = rand_range(0.003,  0.015);
            impulse = (t >= ts) .* (B * exp(-(t - ts) / tau));
            sig = base + impulse;

        case 'Oscillatory_Transient'
            B   = rand_range(0.2, 0.5) * A;
            tau = rand_range(0.001, 0.004);
            fn  = rand_range(300, 1500);
            ts  = rand_range(0.003, 0.015);
            osc = (t >= ts) .* (B * exp(-(t - ts) / tau) .* sin(2*pi*fn*(t - ts)));
            sig = base + osc;

        case 'Harmonics'
            h3 = rand_range(0.05, 0.20);
            h5 = rand_range(0.03, 0.15);
            h7 = rand_range(0.01, 0.10);
            sig = A * ( sin(2*pi*f0*t) ...
                      + h3 * sin(2*pi*3*f0*t) ...
                      + h5 * sin(2*pi*5*f0*t) ...
                      + h7 * sin(2*pi*7*f0*t) );

        case 'Flicker'
            af = rand_range(0.05, 0.15);
            ff = rand_range(8, 25);
            sig = A * (1 + af * sin(2*pi*ff*t)) .* sin(2*pi*f0*t);

        case 'Notch'
            K = rand_range(0.1, 0.4);
            sig = apply_notch(base, K, length(t));

        case 'Sag_Harmonics'
            alpha = rand_range(0.1, 0.9);
            amp = A * (1 - alpha);
            h3 = rand_range(0.05, 0.20);
            h5 = rand_range(0.03, 0.15);
            h7 = rand_range(0.01, 0.10);
            sig = amp * ( sin(2*pi*f0*t) ...
                        + h3 * sin(2*pi*3*f0*t) ...
                        + h5 * sin(2*pi*5*f0*t) ...
                        + h7 * sin(2*pi*7*f0*t) );

        case 'Sag_Flicker'
            alpha = rand_range(0.1, 0.9);
            af = rand_range(0.05, 0.15);
            ff = rand_range(8, 25);
            sagBase  = A * (1 - alpha) * sin(2*pi*f0*t);
            envelope = 1 + af * sin(2*pi*ff*t);
            sig = sagBase .* envelope;

        case 'Sag_Oscillatory'
            alpha = rand_range(0.1, 0.9);
            sagBase = A * (1 - alpha) * sin(2*pi*f0*t);
            B   = rand_range(0.2, 0.5) * A;
            tau = rand_range(0.001, 0.004);
            fn  = rand_range(300, 1500);
            ts  = rand_range(0.003, 0.015);
            osc = (t >= ts) .* (B * exp(-(t - ts) / tau) .* sin(2*pi*fn*(t - ts)));
            sig = sagBase + osc;

        case 'Swell_Harmonics'
            alpha = rand_range(0.1, 0.7);
            amp = A * (1 + alpha);
            h3 = rand_range(0.05, 0.20);
            h5 = rand_range(0.03, 0.15);
            h7 = rand_range(0.01, 0.10);
            sig = amp * ( sin(2*pi*f0*t) ...
                        + h3 * sin(2*pi*3*f0*t) ...
                        + h5 * sin(2*pi*5*f0*t) ...
                        + h7 * sin(2*pi*7*f0*t) );

        case 'Swell_Flicker'
            alpha = rand_range(0.1, 0.7);
            af = rand_range(0.05, 0.15);
            ff = rand_range(8, 25);
            swBase   = A * (1 + alpha) * sin(2*pi*f0*t);
            envelope = 1 + af * sin(2*pi*ff*t);
            sig = swBase .* envelope;

        case 'Swell_Oscillatory'
            alpha = rand_range(0.1, 0.7);
            swBase = A * (1 + alpha) * sin(2*pi*f0*t);
            B   = rand_range(0.2, 0.5) * A;
            tau = rand_range(0.001, 0.004);
            fn  = rand_range(300, 1500);
            ts  = rand_range(0.003, 0.015);
            osc = (t >= ts) .* (B * exp(-(t - ts) / tau) .* sin(2*pi*fn*(t - ts)));
            sig = swBase + osc;

        case 'Harmonics_Flicker'
            h3 = rand_range(0.05, 0.20);
            h5 = rand_range(0.03, 0.15);
            h7 = rand_range(0.01, 0.10);
            af = rand_range(0.05, 0.15);
            ff = rand_range(8, 25);
            harm = A * ( sin(2*pi*f0*t) ...
                       + h3 * sin(2*pi*3*f0*t) ...
                       + h5 * sin(2*pi*5*f0*t) ...
                       + h7 * sin(2*pi*7*f0*t) );
            envelope = 1 + af * sin(2*pi*ff*t);
            sig = harm .* envelope;

        case 'Harmonics_Notch'
            h3 = rand_range(0.05, 0.20);
            h5 = rand_range(0.03, 0.15);
            h7 = rand_range(0.01, 0.10);
            K  = rand_range(0.1, 0.4);
            harm = A * ( sin(2*pi*f0*t) ...
                       + h3 * sin(2*pi*3*f0*t) ...
                       + h5 * sin(2*pi*5*f0*t) ...
                       + h7 * sin(2*pi*7*f0*t) );
            sig = apply_notch(harm, K, length(t));

        otherwise
            error('Unknown class: %s', cls);
    end
end

function out = apply_notch(sig, K, n)
% Apply 4 (1-K)-deep notches at evenly-spaced positions.
    out = sig;
    positions = round([0.20, 0.40, 0.60, 0.80] * n);
    for p = positions
        lo = max(1, p);
        hi = min(n, p + 1);
        out(lo:hi) = out(lo:hi) * (1 - K);
    end
end

function out = add_awgn(sig, snr_db)
% Add AWGN at the specified SNR (dB).
    p_signal = mean(sig.^2);
    noise_std = sqrt(p_signal / 10^(snr_db/10));
    out = sig + noise_std * randn(size(sig));
end

function v = rand_range(a, b)
    v = a + (b - a) * rand;
end
