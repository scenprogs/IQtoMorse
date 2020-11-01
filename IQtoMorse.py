from matplotlib.ticker import FuncFormatter
from scipy import signal
from scipy.signal import hilbert
import argparse
import datetime
import os.path
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.markers as markers
import matplotlib.pyplot as plt
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": "\n".join([
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{times}",
    ]),
})

def between(value, range):
    if range[0] <= value <= range[1]:
        return True
    else:
        return False


def butter_lowpass_filter(samples, fc, fs, order=3):
    # Normalize the frequency
    w = fc / (fs / 2)
    # Get the filter coefficients
    y, x = signal.butter(order, w, btype='lowpass', analog=False)
    # Filter
    z = signal.filtfilt(y, x, samples)
    return z


def butter_bandpass_filter(samples, f_lowcut, f_highcut, fs, order=3):
    nyq = 0.5 * fs
    low = f_lowcut / nyq
    high = f_highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandpass')
    y = signal.filtfilt(b, a, samples)
    return y


def find_peaks(function_x, function_y, threshold, n=5, m=3):
    """Seach for peaks"""
    index = 0
    peaks = []

    while index < len(function_x):

        # Search for points which are above a certain threshold
        if function_y[index] >= threshold:

            count = 0
            count_below = 0
            next_index = index + 1
            if next_index >= len(function_x): break

            start = function_x[index]

            # Count subsequent points which are (also) above threshold
            while function_y[next_index] >= threshold or (count >= n and count_below < m):

                # Allow m values below threshold in a sequence
                if count >= n and function_y[next_index] < threshold:
                    count_below += 1
                else:
                    # reset count below
                    count_below = 0

                count += 1

                if next_index + 1 >= len(function_y) - 1:
                    break
                else:
                    next_index = next_index + 1

            # If count is above n peak is accepted
            if count >= n:
                end = function_x[next_index]
                peaks.append((start, end))

            index = next_index

        else:
            index += 1

    return peaks


def group_peaks(breaks_length, short_break_range, medium_break_range):
    """Group peaks based on the distance (breaks) between peaks"""

    sentence = []
    word = []
    index = 0

    # Iterate breaks & find peaks which belong to one morse symbol (breaks which are within short_break_range)
    while index < len(breaks_length):

        start_index = index
        end_index = index
        last = False

        # Search for groups of short breaks
        while between(breaks_length[end_index], short_break_range):
            # end_index += 1
            if end_index >= len(breaks_length) - 1:
                last = True
                break
            else:
                end_index += 1

        if last and between(breaks_length[end_index], short_break_range):
            word.append((start_index, end_index + 1))
        else:
            word.append((start_index, end_index))

        if breaks_length[end_index] > medium_break_range[1]:
            if len(word) > 0:
                sentence.append(word)
                word = []

        index = end_index + 1

    # In case word wasn't attached to sentence
    if len(word) > 0:
        sentence.append(word)

    return sentence


def peaks_to_morse(grouped_peaks, peak_length_divider):
    """Translate grouped peaks into morse dots (dih's) and lines (dah's)"""

    morse_sentence = []

    for group in grouped_peaks:

        morse_word = []

        for peak_indexes in group:

            morse_symbol = ''

            for peak_index in range(peak_indexes[0], peak_indexes[1] + 1):

                peak = peaks[peak_index]
                peak_length = peak[1] - peak[0]

                if peak_length >= peak_length_divider:
                    morse_symbol += '-'
                else:
                    morse_symbol += '.'

            morse_word.append(morse_symbol)

        morse_sentence.append(morse_word)

    return morse_sentence


def morse_to_chars(morse_sentence):
    """Translate morse dots and lines into characters"""

    sentence = []

    for morse_word in morse_sentence:

        word = []

        for morse_symbol in morse_word:
            if morse_code.__contains__(morse_symbol):
                word.append(morse_code[morse_symbol])
            else:
                word.append('_')

        sentence.append(word)

    return sentence


def morse_to_string(morse_sentence):
    """Translate morse dots and lines into string"""

    sentence = ''
    first = True

    for morse_word in morse_sentence:

        word = ''

        for morse_symbol in morse_word:
            if morse_code.__contains__(morse_symbol):
                word += morse_code[morse_symbol]
            else:
                word += '_'

        if first:
            sentence += word
            first = False
        else:
            sentence += ' ' + word

    return sentence


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle


morse_code = {
    '-.-.--': '!',
    '...-..-': '$',
    '.----.': "'",
    '-.--.': '(',
    '-.--.-': ')',
    '--..--': ',',
    '-....-': '-',
    '.-.-.-': '.',
    '-..-.': '/',
    '-----': '0',
    '.----': '1',
    '..---': '2',
    '...--': '3',
    '....-': '4',
    '.....': '5',
    '-....': '6',
    '--...': '7',
    '---..': '8',
    '----.': '9',
    '---...': ':',
    '-.-.-.': ';',
    '.-.-.': '>',
    '.-...': '<',
    '....--': '{',
    '..-.-': '&',
    '...-.-': '%',
    '...-.': '}',
    '-...-': '=',
    '..--..': '?',
    '.--.-.': '@',
    '.-': 'A',
    '-...': 'B',
    '-.-.': 'C',
    '-..': 'D',
    '.': 'E',
    '..-.': 'F',
    '--.': 'G',
    '....': 'H',
    '..': 'I',
    '.---': 'J',
    '-.-': 'K',
    '.-..': 'L',
    '--': 'M',
    '-.': 'N',
    '---': 'O',
    '.--.': 'P',
    '--.-': 'Q',
    '.-.': 'R',
    '...': 'S',
    '-': 'T',
    '..-': 'U',
    '...-': 'V',
    '.--': 'W',
    '-..-': 'X',
    '-.--': 'Y',
    '--..': 'Z',
    '.-..-.': '\\',
    '..--.-': '_',
    '.-.-': '~',
    '_': ' ',
    '_': '\n'
}

# Plotting disabled at default
enable_plot = False

# Take timestamp for runtime estimation
dt_start = datetime.datetime.now()

# Build arguments
parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('file', metavar="file", type=lambda x: is_valid_file(parser, x), help="input i/q raw file")
required.add_argument('fs', metavar='fs', type=int, nargs='+', help='sample frequency')
optional.add_argument('-fc', metavar='fc', type=float, nargs='+', help='cutoff frequency')
optional.add_argument("-plot", action="store_true")
args = parser.parse_args()

# Import arguments
f_samplerate = args.fs[0]
enable_plot = args.plot

# Import raw I/Q samples
samples = []
samples = np.fromfile(args.file, dtype="float32")
samples = samples[0::2] + 1j * samples[1::2]
data = np.real(samples)
n = len(samples)
t = len(samples) / f_samplerate

print(f"Imported sample with {n} vectors which belongs to a {n / f_samplerate}s ({(n / f_samplerate) * 1000}ms) record")

# (1) Spectral analysis

# Transform given time domain signal into frequency domain with FFT. Applying FFT onto real signal gives us
# the frequency spectrum of our signal. While a morse signal (Continous Wave, CW) has no bandwidth we are
# looking after a single frequency in spectrum with largest magnitude.

# Apply FFT to our signal
X_f = abs(np.fft.fft(samples))
# Calculate frequencies
freq = (f_samplerate / 2) * np.linspace(0, 1, n // 2)
# Only take the positive frequencies
xl_m = (2 / 1) * abs(X_f[0:np.size(freq)])
# Take the frequency from spectrum with largest magnitude
f_peak = freq[xl_m.argmax()]
print('Identified frequency of morse keying at ' + str(f_peak) + ' Hz')

if args.fc:
    f_peak = 0.0 + args.fc[0]

# (2) Low pass filter
# Apply low pass butterworth filter
f_offset = 1
low_passed = butter_lowpass_filter(data, f_peak + f_offset, f_samplerate)

# (3) Generating amplitude envelope
# Apply hilbert transform
analytic_signal = hilbert(low_passed)
# Keep positive envelope
amplitude_envelope = np.abs(analytic_signal)

# Determine noise level based on median of hilbert transform deviation
hist_y, hist_x = np.histogram(amplitude_envelope, bins=100)
noise_median = hist_x[hist_y.tolist().index(hist_y.max())]

# Calculate mean of amplitude envelope
amplitude_envelope_mean = np.mean(amplitude_envelope)

# Find peaks
peaks = find_peaks(np.arange(len(amplitude_envelope)), amplitude_envelope, amplitude_envelope_mean, 500, 0)
print(f"Peaks:")
print(f"\t{len(peaks)} peaks found")

# Calculate peak length
peak_lengths = []
for p in peaks:
    peak_lengths.append(np.abs(p[1] - p[0]))

peak_length_mean = np.mean(peak_lengths)

print(
    f"\tMinimum peak length is {np.min(peak_lengths).round(2)} ({((np.min(peak_lengths) / f_samplerate) * 1000).round(2)}ms)")
print(
    f"\tAverage peak length is {np.mean(peak_lengths).round(2)} ({((np.mean(peak_lengths) / f_samplerate) * 1000).round(2)}ms)")
print(
    f"\tMaximum peak length is {np.max(peak_lengths).round(2)} ({((np.max(peak_lengths) / f_samplerate) * 1000).round(2)}ms)")

# Examine breaks
breaks = []
low_boundary = 0
for p in peaks:
    if low_boundary != 0:
        breaks.append((low_boundary, p[0] - 1))
    low_boundary = p[1] + 1

# Examine lengths of breaks
breaks_length = []
for previous, current in zip(peaks, peaks[1:]):
    breaks_length.append(np.abs(previous[1] - current[0]))

breaks_length = []
for previous, current in zip(peaks, peaks[1:]):
    breaks_length.append(np.abs(previous[1] - current[0]))

hist_y, hist_x = np.histogram(breaks_length, bins=100)
short_break_median = np.median(breaks_length)
short_medium_break_divider = (short_break_median * 3) * 2/3
short_break_range = (0, short_medium_break_divider)
medium_long_break_divider = (short_break_median * 6) * 2/3
medium_break_range = (short_medium_break_divider + 1, medium_long_break_divider)

print(f"\tShort break range is {short_break_range}")
print(f"\tMedium break range is {medium_break_range}")

print(f"Translation:")

# Group peaks
grouped_peaks = group_peaks(breaks_length, short_break_range, medium_break_range)
print(grouped_peaks)

# Translate peaks to dots and lines
peak_length_divider = np.max(peak_lengths) * 0.66
morse = peaks_to_morse(grouped_peaks, peak_length_divider)
print(f"\t{morse}")

# Translate dots and lines to letters
morse_chars = morse_to_chars(morse)
print(f"\t{morse_chars}")
morse_string = morse_to_string(morse)
print(f"\t{morse_string}")

# Calculate runtime
dt_end = datetime.datetime.now()
dt_delta = dt_end - dt_start
print(f"\t{dt_delta}s ({int(dt_delta.total_seconds() * 1000)}ms)")

# Plot if enabled
if enable_plot:
    fig = plt.figure("IQtoMorse - Analytic morse decoder", tight_layout=True, figsize=(16, 9))
    gs = gridspec.GridSpec(4, 3)

    x_axis = np.arange(n)

    ''' Row 1 Col 1 : Real signal '''
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(data, label="Real signal", color='lightgrey')
    ax.set_ylabel('amplitude')
    ax.set_xlabel('time (ms)')
    formatter = FuncFormatter(lambda x, pos: f"{(x/f_samplerate*1000).round(2)}" % (x))
    ax.xaxis.set_major_formatter(formatter)
    ax.legend(loc="upper left")
    ax.text(0.5, 0.5, f"Given sample frequency: {f_samplerate} Hz\n Number of samples in record: {n}\nTotal length of record: {t*1000} ms", fontsize=14, bbox=dict(boxstyle="round, pad=0.5", facecolor='white', alpha=0.5), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    ''' Row 1 Col 2 : PSD of real signal '''
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(freq, 10 * np.log10(xl_m), label="Spectrum of real signal", color='grey');
    ax.set_ylabel('magnitude (dB)')
    ax.set_xlabel('frequency (Hz)')
    ax.legend(loc="upper right")

    ''' Row 1 Col 3 : PSD of real signal (zoomed in)'''
    ax = fig.add_subplot(gs[0, 2])
    ax.axvline(f_peak, color='black', linestyle='dashed', linewidth=1, label=f"Cutoff frequency ({round(f_peak, 2)} Hz)")
    ax.plot(freq, xl_m, label="Spectrum of real signal", color='grey');
    ax.set_ylabel('magnitude')
    ax.set_xlabel('frequency (Hz)')
    ax.axvspan(f_peak - 1, f_peak + 1, alpha=0.1, color='green')
    ax.set_xlim([0, f_peak * 2])
    ax.legend(loc="upper right")

    ''' Row 2+3 Col 1-4 '''
    # Plot functions
    ax = fig.add_subplot(gs[1:-1, :])
    # ax.set_facecolor('black')
    ax.set_ylabel('amplitude')
    ax.set_xlabel('time (ms)')
    ax.xaxis.set_major_formatter(formatter)
    ax.plot(low_passed, '-', color='red', label=f"Low passed")

    # Plot peaks
    amplitude_envelope_max = np.max(amplitude_envelope)

    for idx, p in enumerate(peaks):
        ax.plot(p[0], amplitude_envelope[p[0]], ">", color="black")
        ax.plot(p[1], amplitude_envelope[p[1]], "<", color="black")
        x_peaks = np.arange(p[0], p[1])
        ax.plot(x_peaks, amplitude_envelope[x_peaks], linewidth=5, color="lime", alpha=0.5)
        ax.plot(x_peaks, [amplitude_envelope_mean] * len(x_peaks), linewidth=5, color="lime")
        ax.annotate(f"{((peak_lengths[idx] / f_samplerate) * 1000).round(2)} ms",
                    ((p[0] + p[1]) / 2, amplitude_envelope_mean + (1/5) * amplitude_envelope_mean),
                    textcoords="offset points", xytext=(0, 0), ha='center', size=10,
                    bbox=dict(boxstyle="round, pad=0.25", fc='white', alpha=0.5))

    # Plot breaks
    for idx, b in enumerate(breaks):
        x_breaks = np.arange(b[0], b[1])
        ax.plot(x_breaks, amplitude_envelope[x_breaks], linestyle=(0, (3, 1)), linewidth=5, color="grey", alpha=0.5)
        ax.plot(x_breaks, [amplitude_envelope_mean] * len(x_breaks), linestyle=(0, (3, 1)), linewidth=5, color="grey")
        ax.annotate(f"{((breaks_length[idx] / f_samplerate) * 1000).round(2)} ms",
                    ((b[0] + b[1]) / 2, amplitude_envelope_mean + (-1/5) * amplitude_envelope_mean),
                    textcoords="offset points", xytext=(0, 0), ha='center', size=10,
                    bbox=dict(boxstyle="round, pad=0.25", fc='white', alpha=0.5))

    ax.plot(x_axis, amplitude_envelope, 'b-', label=f"Amplitude envelope (hilbert transformation)")
    ax.plot(x_axis, [noise_median] * len(samples), ':', color="orange", label=f"Amplitude envelope modal (noise level)")
    ax.plot(x_axis, [amplitude_envelope_mean] * len(samples), '--', color="green", linewidth=1, label=f"Amplitude envelope mean (peak level)")

    # Plot translated letters below corresponding peaks
    for idx1, word in enumerate(grouped_peaks):
        for idx2, letter in enumerate(word):
            letter_range_x = np.arange(peaks[letter[0]][0], peaks[letter[1]][1])
            ax.plot(peaks[letter[0]][0], -amplitude_envelope_mean, marker=markers.CARETRIGHT, color='black', markersize=10)
            ax.plot(peaks[letter[1]][1], -amplitude_envelope_mean, marker=markers.CARETLEFT, color='black', markersize=10)
            ax.plot(letter_range_x, [-amplitude_envelope_mean] * len(letter_range_x), 'k--', linewidth=1)
            ax.annotate(morse_chars[idx1][idx2], ((peaks[letter[0]][0]+peaks[letter[1]][1])/2, -amplitude_envelope_mean + (-1/4) * amplitude_envelope_mean),
                        textcoords="offset points", xytext=(0, 0), ha='center', size=12,
                        bbox=dict(boxstyle="round, pad=0.5", fc='white', alpha=0.75))
            ax.axvspan(peaks[letter[0]][0], peaks[letter[1]][1], alpha=0.1, color='black')

    ax.text(0.5, 0.1, morse_string, fontsize=16, bbox=dict(boxstyle="round, pad=1", facecolor='white', alpha=0.5), horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
    ax.legend(loc="upper left")

    ''' Row 4 - Col 1 : AMPLITUDE ENVELOPE'''
    ax = fig.add_subplot(gs[3, 0])
    plt.hist(amplitude_envelope, bins=100, histtype='stepfilled', density=True, color="blue")
    ax.axvline(amplitude_envelope_mean, color='y', linestyle='dashed', linewidth=1.5, label='Amplitude envelope mean (peak level)')
    ax.axvline(noise_median, color='orange', linestyle='dashed', linewidth=1.5, label='Amplitude envelope modal (noise level)')
    ax.legend(loc="upper right")
    ax.set_title("Histogram of amplitude envelope")

    ''' Row 4 - Col 2 : PEAKS '''
    ax = fig.add_subplot(gs[3, 1])
    ax.axvline(peak_length_divider, color='black', linestyle=':', linewidth=1.5, label='Peak length divider')
    ax.axvspan(0, peak_length_divider, alpha=0.1, color='grey', label='Short peak range')
    ax.axvspan(peak_length_divider + 1, np.max(peak_lengths), alpha=0.1, color='blue', label='Long peak range')
    ax.hist(peak_lengths, bins=50, histtype='bar', color="lime")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(formatter)
    ax.set_title("Histogram of peak lengths")
    ax.set_xlabel('time (ms)')

    ''' Row 4 - Col 3 : BREAKS '''
    ax = fig.add_subplot(gs[3, 2])
    ax.axvline(short_break_median, color='orange', linestyle='-', linewidth=2, label='Break length median')
    ax.axvline(short_medium_break_divider, color='black', linestyle=':', linewidth=1.5, label='Divider short/medium breaks')
    ax.axvline(medium_long_break_divider, color='blue', linestyle=':', linewidth=1.5, label='Divider medium/long breaks')
    ax.axvspan(short_break_range[0], short_break_range[1], alpha=0.1, color='grey', label='Short break range')
    ax.axvspan(medium_break_range[0], medium_break_range[1], alpha=0.1, color='blue', label='Medium break range')
    ax.axvspan(medium_break_range[1], np.max(breaks_length), alpha=0.1, color='green', label='Long break range')
    ax.hist(breaks_length, bins=50, histtype='stepfilled', color="grey")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(formatter)
    ax.set_title("Histogram of break lengths")
    ax.set_xlabel('time (ms)')

    fig.tight_layout()
    plt.show()
