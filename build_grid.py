import numpy as np

from config import (
    N_BANDS,
    TARGET_SLOTS,
    MAX_SLOTS
)


def estimate_slot_width(toa, target_slots=TARGET_SLOTS):

    toa = np.asarray(toa, dtype=np.float64)
    toa = toa[np.isfinite(toa)]

    if len(toa) == 0:
        raise ValueError("No valid TOA values found.")

    span = toa.max() - toa.min()

    if span <= 0:
        raise ValueError("TOA span is zero or negative.")

    return span / target_slots


def print_memory_estimate(n_bands, n_slots):

    # occupancy uint8
    occupancy_bytes = n_bands * n_slots

    # amplitude float32
    amplitude_bytes = n_bands * n_slots * 4

    # PW float32
    pw_bytes = n_bands * n_slots * 4

    # AoA float32
    aoa_bytes = n_bands * n_slots * 4

    total = (
        occupancy_bytes
        + amplitude_bytes
        + pw_bytes
        + aoa_bytes
    )

    mb = total / (1024 ** 2)
    gb = total / (1024 ** 3)

    print("\n========== GRID MEMORY ESTIMATE ==========")
    print("Bands:", n_bands)
    print("Time slots:", n_slots)
    print("Grid shape:", (n_bands, n_slots))
    print(f"Approx memory: {mb:.2f} MB ({gb:.3f} GB)")
    print("==========================================\n")


def build_grid(
    df,
    freq_col,
    toa_col,
    pw_col=2,
    aoa_col=3,
    amp_col=4,
    slot_width=None
):

    # Validate columns
    for col, name in [
        (freq_col, "frequency"),
        (toa_col, "TOA"),
        (pw_col, "PW"),
        (aoa_col, "AoA"),
        (amp_col, "amplitude")
    ]:
        if col not in df.columns:
            raise ValueError(
                f"{name} column '{col}' not found."
            )

    # Get values
    freq = df[freq_col].to_numpy(dtype=np.float64)
    toa = df[toa_col].to_numpy(dtype=np.float64)
    pw = df[pw_col].to_numpy(dtype=np.float64)
    aoa = df[aoa_col].to_numpy(dtype=np.float64)
    amp = df[amp_col].to_numpy(dtype=np.float64)

    # Remove invalid rows
    valid = (
        np.isfinite(freq)
        & np.isfinite(toa)
        & np.isfinite(pw)
        & np.isfinite(aoa)
        & np.isfinite(amp)
        & (pw >= 0)
    )

    freq = freq[valid]
    toa = toa[valid]
    pw = pw[valid]
    aoa = aoa[valid]
    amp = amp[valid]

    if len(freq) == 0:
        raise ValueError("No valid PDWs remain.")

    print("Valid PDWs:", len(freq))

    # Slot width
    if slot_width is None:
        slot_width = estimate_slot_width(toa)

    if slot_width <= 0:
        raise ValueError("slot_width must be > 0.")

    # Relative time
    toa_min = toa.min()
    rel_toa = toa - toa_min

    # Pulse END = TOA + PW
    pulse_end = rel_toa + pw

    # Number of slots
    n_slots = int(
        np.ceil(pulse_end.max() / slot_width)
    ) + 1

    if n_slots > MAX_SLOTS:
        raise MemoryError(
            f"Grid requires {n_slots:,} slots, "
            f"but MAX_SLOTS is {MAX_SLOTS:,}."
        )

    # Frequency bands
    freq_min = freq.min()
    freq_max = freq.max()
    freq_range = freq_max - freq_min

    if freq_range == 0:

        bands = np.zeros(
            len(freq),
            dtype=np.int32
        )

    else:

        bands = (
            (freq - freq_min)
            / freq_range
            * N_BANDS
        ).astype(np.int32)

        bands = np.clip(
            bands,
            0,
            N_BANDS - 1
        )

    # Start/end slots for each pulse
    start_slots = np.floor(
        rel_toa / slot_width
    ).astype(np.int64)

    end_slots = np.ceil(
        pulse_end / slot_width
    ).astype(np.int64)

    start_slots = np.clip(
        start_slots,
        0,
        n_slots - 1
    )

    end_slots = np.clip(
        end_slots,
        0,
        n_slots - 1
    )

    # Create grids
    occupancy_grid = np.zeros(
        (N_BANDS, n_slots),
        dtype=np.uint8
    )

    amplitude_grid = np.full(
        (N_BANDS, n_slots),
        -np.inf,
        dtype=np.float32
    )

    pw_grid = np.zeros(
        (N_BANDS, n_slots),
        dtype=np.float32
    )

    aoa_grid = np.full(
        (N_BANDS, n_slots),
        np.nan,
        dtype=np.float32
    )

    # Put pulses into grid
    for b, s0, s1, a, p, angle in zip(
        bands,
        start_slots,
        end_slots,
        amp,
        pw,
        aoa
    ):

        if s1 < s0:
            continue

        for s in range(s0, s1 + 1):

            occupancy_grid[b, s] = 1

            # Keep strongest signal
            if a > amplitude_grid[b, s]:

                amplitude_grid[b, s] = a
                pw_grid[b, s] = p
                aoa_grid[b, s] = angle

    print("\n========== GRID PARAMETERS ==========")
    print("Frequency min:", freq_min)
    print("Frequency max:", freq_max)
    print("Original TOA min:", toa_min)
    print("Slot width:", slot_width)
    print("Number of slots:", n_slots)
    print("=====================================\n")

    print_memory_estimate(
        N_BANDS,
        n_slots
    )

    print("Final grid shape:", occupancy_grid.shape)
    print("Occupied cells:", int(occupancy_grid.sum()))
    print("Total cells:", occupancy_grid.size)

    return (
        occupancy_grid,
        amplitude_grid,
        pw_grid,
        aoa_grid,
        slot_width
    )