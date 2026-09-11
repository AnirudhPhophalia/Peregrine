import numpy as np

from config import (
    FREQ_MAX_MHZ,
    FREQ_MIN_MHZ,
    MAX_PW_US,
    MAX_SLOTS,
    MIN_PW_US,
    N_BANDS,
    SCAN_STEP_MHZ,
    SIMULATION_DURATION_US,
    TARGET_SLOTS,
)


def print_memory_estimate(n_bands, n_slots):
    total = (
        n_bands * n_slots * 1
        + n_bands * n_slots * 4
        + n_bands * n_slots * 4
        + n_bands * n_slots * 4
    )

    print("\n========== GRID MEMORY ESTIMATE ==========")
    print("Bands:", n_bands)
    print("Time slots:", n_slots)
    print("Grid shape:", (n_bands, n_slots))
    print(f"Approx memory: {total / 1024**2:.2f} MB")
    print("==========================================\n")


def estimate_slot_width(toa, target_slots=TARGET_SLOTS):
    toa = np.asarray(toa, dtype=np.float64)
    toa = toa[np.isfinite(toa)]

    if toa.size == 0:
        return SIMULATION_DURATION_US / target_slots

    span = min(
        float(np.max(toa)),
        SIMULATION_DURATION_US,
    )

    if span <= 0:
        return SIMULATION_DURATION_US / target_slots

    return max(
        span / target_slots,
        SIMULATION_DURATION_US / target_slots,
    )


def build_grid(
    df,
    freq_col=1,
    toa_col=0,
    pw_col=2,
    aoa_col=3,
    amp_col=4,
    slot_width=None,
):
    required = {
        freq_col: "frequency",
        toa_col: "TOA",
        pw_col: "PW",
        aoa_col: "AoA",
        amp_col: "amplitude",
    }

    for col, name in required.items():
        if col not in df.columns:
            raise ValueError(
                f"{name} column '{col}' not found."
            )

    freq = df[freq_col].to_numpy(np.float64)
    toa = df[toa_col].to_numpy(np.float64)
    pw = df[pw_col].to_numpy(np.float64)
    aoa = df[aoa_col].to_numpy(np.float64)
    amp = df[amp_col].to_numpy(np.float64)

    valid = (
        np.isfinite(freq)
        & np.isfinite(toa)
        & np.isfinite(pw)
        & np.isfinite(aoa)
        & np.isfinite(amp)
    )

    freq = freq[valid]
    toa = toa[valid]
    pw = pw[valid]
    aoa = aoa[valid]
    amp = amp[valid]

    # Dataset frequency is MHz.
    valid_freq = (
        (freq >= FREQ_MIN_MHZ)
        & (freq <= FREQ_MAX_MHZ)
    )

    freq = freq[valid_freq]
    toa = toa[valid_freq]
    pw = pw[valid_freq]
    aoa = aoa[valid_freq]
    amp = amp[valid_freq]

    valid_pw = (
        (pw >= MIN_PW_US)
        & (pw <= MAX_PW_US)
    )

    freq = freq[valid_pw]
    toa = toa[valid_pw]
    pw = pw[valid_pw]
    aoa = aoa[valid_pw]
    amp = amp[valid_pw]

    valid_time = (
        (toa >= 0)
        & (toa < SIMULATION_DURATION_US)
    )

    freq = freq[valid_time]
    toa = toa[valid_time]
    pw = pw[valid_time]
    aoa = aoa[valid_time]
    amp = amp[valid_time]

    if len(freq) == 0:
        raise ValueError(
            "No valid PDWs remain after filtering."
        )

    if slot_width is None:
        slot_width = (
            SIMULATION_DURATION_US / TARGET_SLOTS
        )

    n_slots = int(
        np.ceil(
            SIMULATION_DURATION_US
            / slot_width
        )
    )

    if n_slots > MAX_SLOTS:
        raise MemoryError(
            f"Simulation requires {n_slots:,} slots, "
            f"but MAX_SLOTS is {MAX_SLOTS:,}."
        )

    # 500 MHz receiver bands.
    # Band 0 = 0-500 MHz, band 1 = 500-1000 MHz, etc.
    bands = np.floor(
        (freq - FREQ_MIN_MHZ)
        / SCAN_STEP_MHZ
    ).astype(np.int32)

    bands = np.clip(
        bands,
        0,
        N_BANDS - 1,
    )

    start_slots = np.floor(
        toa / slot_width
    ).astype(np.int64)

    end_slots = np.floor(
        (toa + pw) / slot_width
    ).astype(np.int64)

    start_slots = np.clip(
        start_slots,
        0,
        n_slots - 1,
    )

    end_slots = np.clip(
        end_slots,
        0,
        n_slots - 1,
    )

    occupancy_grid = np.zeros(
        (N_BANDS, n_slots),
        dtype=np.uint8,
    )

    amplitude_grid = np.full(
        (N_BANDS, n_slots),
        -np.inf,
        dtype=np.float32,
    )

    pw_grid = np.zeros(
        (N_BANDS, n_slots),
        dtype=np.float32,
    )

    aoa_grid = np.full(
        (N_BANDS, n_slots),
        np.nan,
        dtype=np.float32,
    )

    for (
        band,
        start,
        end,
        amplitude,
        pulse_width,
        angle,
    ) in zip(
        bands,
        start_slots,
        end_slots,
        amp,
        pw,
        aoa,
    ):
        if end < start:
            continue

        end = min(
            end,
            n_slots - 1,
        )

        sl = slice(
            start,
            end + 1,
        )

        occupancy_grid[
            band,
            sl,
        ] = 1

        current = amplitude_grid[
            band,
            sl,
        ]

        stronger = amplitude > current
        idx = np.flatnonzero(stronger)

        if idx.size:
            band_idx = (
                np.arange(start, end + 1)[idx]
            )

            amplitude_grid[
                band,
                band_idx,
            ] = amplitude

            pw_grid[
                band,
                band_idx,
            ] = pulse_width

            aoa_grid[
                band,
                band_idx,
            ] = angle

    print("\n========== RF GRID ==========")
    print(
        f"Frequency range: "
        f"{FREQ_MIN_MHZ:.0f} - "
        f"{FREQ_MAX_MHZ:.0f} MHz"
    )
    print(
        f"Band width: {SCAN_STEP_MHZ:.0f} MHz"
    )
    print(
        f"Slot duration: {slot_width:.3f} us"
    )
    print(
        f"Simulation duration: "
        f"{SIMULATION_DURATION_US / 1e6:.2f} s"
    )
    print(
        f"Number of slots: {n_slots:,}"
    )
    print(
        f"Grid shape: {occupancy_grid.shape}"
    )
    print(
        f"Occupied cells: "
        f"{int(occupancy_grid.sum()):,}"
    )
    print("=============================\n")

    print_memory_estimate(
        N_BANDS,
        n_slots,
    )

    return (
        occupancy_grid,
        amplitude_grid,
        pw_grid,
        aoa_grid,
        slot_width,
    )
