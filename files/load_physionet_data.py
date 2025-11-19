""" "
This is a function for loading data from a physionet dataset to csv, and adding timestamp.
"""

import wfdb, pandas as pd, numpy as np
from pathlib import Path

pn_dir = Path("files/physionet.org/files/mhd-effect-ecg-mri/1.0.0")
# f2* (young + old) have BP according to the database docs
records = [f.stem for f in pn_dir.glob("*.hea")]


def myfilt(s: str):
    p = pn_dir / f"{s}.dat"
    return p.is_file()


records = list(filter(myfilt, records))

SAMPLINGS_RATE_GOAL = 10000
N_SAMPLES = 100000


# This creates a folder for saving data to
out_dir = Path("files/mhd-effect-ecg-mri")
out_dir.mkdir(parents=True, exist_ok=True)


def export_bp_csv(record):
    rec = wfdb.rdrecord(pn_dir / record)
    fs: float = rec.fs  # type: ignore

    # decimation = int(fs / SAMPLINGS_RATE_GOAL)
    names = rec.sig_name
    assert hasattr(rec, "p_signal")
    assert isinstance(rec.p_signal, np.ndarray), "Data must be an ndarray"  # type: ignore
    data: np.ndarray = rec.p_signal  # type: ignore
    assert isinstance(data, np.ndarray)
    # Check if .p_signal exists
    assert isinstance(names, list)
    if len(data) > N_SAMPLES:
        data = data[:N_SAMPLES, :]

    t = np.arange(len(data)) / fs
    dict_data: dict[str, np.ndarray] = {"time_s": t}
    for i, n in enumerate(names):
        assert isinstance(n, str)
        assert isinstance(i, int)
        dict_data[n] = data[:, i]
    df = pd.DataFrame(dict_data)
    # df = df.iloc[::decimation].reset_index(drop=True)
    df.to_csv(out_dir / f"{record}.csv", index=False)

    subjectinfo = rec.record_name
    print(f"[ok] {record} -> {out_dir / f'{record}.csv'}")


for i, r in enumerate(records):
    if i > 10:
        break
    export_bp_csv(Path(r))
