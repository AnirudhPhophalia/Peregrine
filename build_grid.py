import numpy as np

from config import (
    N_BANDS,
    FREQ_MIN_GHZ,
    FREQ_MAX_GHZ,
    SIMULATION_DURATION_US,
    SLOT_DURATION_US,
    MAX_SLOTS,
    MIN_PW_US,
    MAX_PW_US
)


# MEMORY ESTIMATE
def print_memory_estimate(
    n_bands,
    n_slots
):

    occupancy_bytes = (
        n_bands * n_slots
    )

    amplitude_bytes = (
        n_bands * n_slots * 4
    )

    pw_bytes = (
        n_bands * n_slots * 4
    )

    aoa_bytes = (
        n_bands * n_slots * 4
    )

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


# BUILD RF GRID
def build_grid(
    df,
    freq_col,
    toa_col,
    pw_col=2,
    aoa_col=3,
    amp_col=4
):

    # Validate columns
    required = {
        freq_col: "frequency",
        toa_col: "TOA",
        pw_col: "PW",
        aoa_col: "AoA",
        amp_col: "amplitude"
    }

    for col, name in required.items():

        if col not in df.columns:

            raise ValueError(
                f"{name} column '{col}' not found."
            )

    # Extract dataset columns
    freq = df[freq_col].to_numpy(
        dtype=np.float64
    )

    toa = df[toa_col].to_numpy(
        dtype=np.float64
    )

    pw = df[pw_col].to_numpy(
        dtype=np.float64
    )

    aoa = df[aoa_col].to_numpy(
        dtype=np.float64
    )

    amp = df[amp_col].to_numpy(
        dtype=np.float64
    )

    # Remove invalid rows
    valid = (
        np.isfinite(freq)
        &
        np.isfinite(toa)
        &
        np.isfinite(pw)
        &
        np.isfinite(aoa)
        &
        np.isfinite(amp)
    )

    freq = freq[valid]
    toa = toa[valid]
    pw = pw[valid]
    aoa = aoa[valid]
    amp = amp[valid]

    # Restrict frequency to receiver spectrum
    valid_freq = (
        (freq >= FREQ_MIN_GHZ)
        &
        (freq <= FREQ_MAX_GHZ)
    )

    freq = freq[valid_freq]
    toa = toa[valid_freq]
    pw = pw[valid_freq]
    aoa = aoa[valid_freq]
    amp = amp[valid_freq]

    # Restrict pulse widths
    valid_pw = (
        (pw >= MIN_PW_US)
        &
        (pw <= MAX_PW_US)
    )

    freq = freq[valid_pw]
    toa = toa[valid_pw]
    pw = pw[valid_pw]
    aoa = aoa[valid_pw]
    amp = amp[valid_pw]

    if len(freq) == 0:

        raise ValueError(
            "No valid PDWs remain after filtering."
        )

    print(
        "Valid PDWs:",
        len(freq)
    )

    # Simulation time
    valid_time = (
        (toa >= 0)
        &
        (toa < SIMULATION_DURATION_US)
    )

    freq = freq[valid_time]
    toa = toa[valid_time]
    pw = pw[valid_time]
    aoa = aoa[valid_time]
    amp = amp[valid_time]

    if len(freq) == 0:

        raise ValueError(
            "No PDWs fall inside the simulation interval."
        )

    # Number of simulation slots
    n_slots = int(
        np.ceil(
            SIMULATION_DURATION_US
            / SLOT_DURATION_US
        )
    )

    if n_slots > MAX_SLOTS:

        raise MemoryError(
            f"Simulation requires "
            f"{n_slots:,} slots, but MAX_SLOTS is "
            f"{MAX_SLOTS:,}."
        )

    # Frequency band mapping
    band_width = (
        FREQ_MAX_GHZ - FREQ_MIN_GHZ
    ) / N_BANDS

    bands = (
        (
            freq - FREQ_MIN_GHZ
        )
        / band_width
    ).astype(np.int32)

    bands = np.clip(
        bands,
        0,
        N_BANDS - 1
    )

    # Time-slot mapping
    start_slots = (
        toa / SLOT_DURATION_US
    ).astype(np.int64)

    # PW determines how many physical time slots the pulse
    # remains active.

    end_slots = (
        (toa + pw)
        / SLOT_DURATION_US
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

    # Allocate grids
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

    # Insert pulses
    for (
        band,
        start,
        end,
        amplitude,
        pulse_width,
        angle
    ) in zip(
        bands,
        start_slots,
        end_slots,
        amp,
        pw,
        aoa
    ):

        if end < start:
            continue

        # Make sure the pulse doesn't extend beyond the
        # simulation.

        end = min(
            end,
            n_slots - 1
        )

        occupancy_grid[
            band,
            start:end + 1
        ] = 1

        # If multiple pulses overlap in the same band,
        # retain the strongest received signal.

        current = amplitude_grid[
            band,
            start:end + 1
        ]

        stronger = (
            amplitude > current
        )

        amplitude_grid[
            band,
            start:end + 1
        ][stronger] = amplitude

        pw_grid[
            band,
            start:end + 1
        ][stronger] = pulse_width

        aoa_grid[
            band,
            start:end + 1
        ][stronger] = angle

    # Diagnostics
    print("\n========== RF GRID ==========")

    print(
        "Frequency range:",
        f"{FREQ_MIN_GHZ} - {FREQ_MAX_GHZ} GHz"
    )
    print(
        "Band width:",
        f"{band_width:.3f} GHz"
    )
    print(
        "Slot duration:",
        f"{SLOT_DURATION_US} us"
    )
    print(
        "Simulation duration:",
        f"{SIMULATION_DURATION_US / 1e6:.2f} s"
    )
    print(
        "Number of slots:",
        f"{n_slots:,}"
    )
    print(
        "Grid shape:",
        occupancy_grid.shape
    )
    print(
        "Occupied cells:",
        int(occupancy_grid.sum())
    )
    print("=============================\n")

    print_memory_estimate(
        N_BANDS,
        n_slots
    )

    return (
        occupancy_grid,
        amplitude_grid,
        pw_grid,
        aoa_grid,
        SLOT_DURATION_US
    )