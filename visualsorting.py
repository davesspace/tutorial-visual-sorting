import time
import wave
import numpy as np
import scipy as sp
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('dark_background')  # comment out for "light" theme
plt.rcParams["font.size"] = 16


class TrackedArray():

    def __init__(self, arr, kind="minimal"):
        self.arr = np.copy(arr)
        self.kind = kind
        self.reset()

    def reset(self):
        self.indices = []
        self.values = []
        self.access_type = []
        self.full_copies = []

    def track(self, key, access_type):
        self.indices.append(key)
        self.values.append(self.arr[key])
        self.access_type.append(access_type)
        if self.kind == "full":
            self.full_copies.append(np.copy(self.arr))

    def GetActivity(self, idx=None):
        if isinstance(idx, type(None)):
            return [(i, op) for (i, op) in zip(self.indices, self.access_type)]
        else:
            return (self.indices[idx], self.access_type[idx])

    def __delitem__(self, key):
        self.track(key, "del")
        self.arr.__delitem__(key)

    def __getitem__(self, key):
        self.track(key, "get")
        return self.arr.__getitem__(key)

    def __setitem__(self, key, value):
        self.arr.__setitem__(key, value)
        self.track(key, "set")

    def __len__(self):
        return self.arr.__len__()

    def __str__(self):
        return self.arr.__str__()

    def __repr__(self):
        return self.arr.__repr__()


def freq_map(x, x_min=0, x_max=1000, freq_min=120, freq_max=1200):
    """ map a value x to a frequency f and return a chunk of that
    frequency for the specificed time dt"""
    return np.interp(x, [x_min, x_max], [freq_min, freq_max])


def freq_sample(freq, dt=1./60., samplerate=44100, oversample=2):
    """Create a sample with a specific freqency {freq} for a specified
    time {dt}"""
    mid_samples = np.int(dt * samplerate)
    pad_samples = np.int((mid_samples*(oversample-1)/2))
    total_samples = mid_samples + 2*pad_samples

    y = np.sin(2 * np.pi * freq * np.linspace(0, dt, total_samples))
    y[:pad_samples] = y[:pad_samples] * np.linspace(0, 1, pad_samples)
    y[- pad_samples:] = y[len(y) - pad_samples:] * \
        np.linspace(1, 0, pad_samples)

    return y


N = 30
FPS = 60
OVERSAMPLE = 2
F_SAMPLE = 44100

arr = np.round(np.linspace(0, 1000, N), 0)
np.random.seed(0)
np.random.shuffle(arr)

arr = TrackedArray(arr, "full")

np.random.seed(0)

# ##############################################
# ########### DEMO 1 - Insertion Sort ##########
# ##############################################
# sorter = "Insertion"
# t0 = time.perf_counter()
# i = 1
# while (i < len(arr)):
#     j = i
#     while ((j > 0) and (arr[j-1] > arr[j])):
#         temp = arr[j-1]
#         arr[j-1] = arr[j]
#         arr[j] = temp
#         j -= 1

#     i += 1
# t_ex = time.perf_counter() - t0
# ##############################################

##############################################
########### DEMO 2 - Quick sort ##############
##############################################
sorter = "Quick"


def quicksort(A, lo, hi):
    if lo < hi:
        p = partition(A, lo, hi)
        quicksort(A, lo, p - 1)
        quicksort(A, p + 1, hi)


def partition(A, lo, hi):
    pivot = A[hi]
    i = lo
    for j in range(lo, hi):
        if A[j] < pivot:
            temp = A[i]
            A[i] = A[j]
            A[j] = temp
            i += 1
    temp = A[i]
    A[i] = A[hi]
    A[hi] = temp
    return i


t0 = time.perf_counter()

quicksort(arr, 0, len(arr)-1)

t_ex = time.perf_counter() - t0
##############################################

print(f"---------- {sorter} Sort ----------")
print(f"Array Sorted in {t_ex*1E3:.1f} ms | {len(arr.full_copies):.0f} "
      f"array access operations were performed")


wav_data = np.zeros(np.int(F_SAMPLE*len(arr.values)*1./FPS), dtype=np.float)
dN = np.int(F_SAMPLE * 1./FPS)  # how many samples is each chunk

for i, value in enumerate(arr.values):

    freq = freq_map(value)

    sample = freq_sample(freq, dt=1./FPS, samplerate=F_SAMPLE,
                         oversample=OVERSAMPLE)

    idx_0 = np.int((i+0.5)*dN - len(sample)/2)
    idx_1 = idx_0 + len(sample)

    try:
        wav_data[idx_0:idx_1] = wav_data[idx_0:idx_1] + sample
    except ValueError:
        print(f"Failed to generate {i:.0f}th index sample")

wav_data = (2**15*(wav_data/np.max(np.abs(wav_data)))).astype(np.int16)

sp.io.wavfile.write(f"{sorter}_sound.wav", F_SAMPLE, wav_data)

fig, ax = plt.subplots(figsize=(16, 8))
container = ax.bar(np.arange(0, len(arr), 1),
                   arr.full_copies[0], align="edge", width=0.8)
fig.suptitle(f"{sorter} sort")
ax.set(xlabel="Index", ylabel="Value")
ax.set_xlim([0, N])
txt = ax.text(0.01, 0.99, "", ha="left", va="top", transform=ax.transAxes)


def update(frame):
    txt.set_text(f"Accesses = {frame}")
    for rectangle, height in zip(container.patches, arr.full_copies[frame]):
        rectangle.set_height(height)
        rectangle.set_color("#1f77b4")

    idx, op = arr.GetActivity(frame)
    if op == "get":
        container.patches[idx].set_color("magenta")
    elif op == "set":
        container.patches[idx].set_color("red")

    fig.savefig(f"frames/{sorter}_frame{frame:05.0f}.png")

    return (txt, *container)


ani = FuncAnimation(fig, update, frames=range(len(arr.full_copies)),
                    blit=True, interval=1000./FPS, repeat=False)
