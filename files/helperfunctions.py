from scipy.signal import butter, sosfilt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 5.2 Hjælpefunktioner: fitler og plot af signalerne
def butter_bandpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = np.array(cutoff) / nyq
    sos = butter(order, normal_cutoff, output="sos", btype="bandpass")
    return sos


def butter_bandpass_filter(data: list | np.ndarray, cutoff:list[float], fs: float, order=2) -> np.ndarray:
    data =np.array(data)

    sos = butter_bandpass(cutoff, fs, order=order)
    data = data - data[0]
    y = sosfilt(sos, data)

    return np.array(y)


def plot_ecg_segment(
    df_seg: pd.DataFrame,
    *,
    y_columns: list[str],
    x_column: str = "time_s",
    use_baseline_correction=True,
    fs: float | int | None = None,
    title="EKG-udsnit",
):
    assert isinstance(y_columns, list | tuple | set)
    assert set(y_columns).issubset(df_seg.columns)

    t = df_seg[x_column].to_numpy()

    ylabel = "Amplitude (a.e.)"
    fig, ax = plt.subplots(figsize=(10, 5))
    for lbl in y_columns:
        y0 = df_seg[lbl].to_numpy()
        if use_baseline_correction and fs is not None:
            y = butter_bandpass_filter(y0, [1.0, 100.0], fs)
        else:
            y = y0
        ax.plot(t, y, label=lbl)

    ax.set_xlabel("Tid (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()