import numpy as np

from config import (
    N_BANDS,
    TARGET_SLOTS,
    MAX_SLOTS
)


def estimate_slot_width(
    toa,
    target_slots=TARGET_SLOTS
):

    toa = np.asarray(
        toa,
        dtype=np.float64
    )

    toa = toa[
        np.isfinite(toa)
    ]

    if len(toa) == 0:
        raise ValueError(
            "No valid TOA values found."
        )

    span = (
        toa.max()
        - toa.min()
    )

    if span <= 0:
        raise ValueError(
            "TOA span is zero or negative."
        )

    return span / target_slots


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

    print("\nGRID MEMORY")
    print(
        f"Bands: {n_bands}"
    )
    print(
        f"Time slots: {n_slots:,}"
    )
    print(
        f"Grid shape: "
        f"({n_bands}, {n_slots:,})"
    )
    print(
        f"Approx memory: "
        f"{mb:.2f} MB "
        f"({gb:.3f} GB)"
    )


def build_grid(
    df,
    freq_col,
    toa_col,
    pw_col=2,
    aoa_col=3,
    amp_col=4,
    slot_width=None
):

    required = [
        (freq_col, "frequency"),
        (toa_col, "TOA"),
        (pw_col, "PW"),
        (aoa_col, "AoA"),
        (amp_col, "amplitude")
    ]

    for col, name in required:

        if col not in df.columns:
            raise ValueError(
                f"{name} column "
                f"'{col}' not found."
            )

    freq = df[
        freq_col
    ].to_numpy(
        dtype=np.float64
    )

    toa = df[
        toa_col
    ].to_numpy(
        dtype=np.float64
    )

    pw = df[
        pw_col
    ].to_numpy(
        dtype=np.float64
    )

    aoa = df[
        aoa_col
    ].to_numpy(
        dtype=np.float64
    )

    amp = df[
        amp_col
    ].to_numpy(
        dtype=np.float64
    )

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
        raise ValueError(
            "No valid PDWs remain."
        )

    print(
        f"Valid PDWs: "
        f"{len(freq):,}"
    )

    if slot_width is None:

        slot_width = estimate_slot_width(
            toa
        )

    if slot_width <= 0:
        raise ValueError(
            "slot_width must be positive."
        )

    toa_min = toa.min()

    rel_toa = toa - toa_min

    pulse_end = (
        rel_toa + pw
    )

    n_slots = int(
        np.ceil(
            pulse_end.max()
            / slot_width
        )
    ) + 1

    if n_slots > MAX_SLOTS:

        raise MemoryError(
            f"Grid requires "
            f"{n_slots:,} slots, "
            f"but MAX_SLOTS is "
            f"{MAX_SLOTS:,}."
        )

    freq_min = freq.min()

    freq_max = freq.max()

    freq_range = (
        freq_max - freq_min
    )

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
        ).astype(
            np.int32
        )

        bands = np.clip(
            bands,
            0,
            N_BANDS - 1
        )

    start_slots = np.floor(
        rel_toa / slot_width
    ).astype(
        np.int64
    )

    end_slots = np.ceil(
        pulse_end / slot_width
    ).astype(
        np.int64
    )

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

        for slot in range(
            start,
            end + 1
        ):

            occupancy_grid[
                band,
                slot
            ] = 1

            if (
                amplitude
                > amplitude_grid[
                    band,
                    slot
                ]
            ):

                amplitude_grid[
                    band,
                    slot
                ] = amplitude

                pw_grid[
                    band,
                    slot
                ] = pulse_width

                aoa_grid[
                    band,
                    slot
                ] = angle

    print("\nGRID PARAMETERS")

    print(
        f"Frequency min: "
        f"{freq_min}"
    )

    print(
        f"Frequency max: "
        f"{freq_max}"
    )

    print(
        f"Frequency range: "
        f"{freq_range}"
    )

    print(
        f"Original TOA minimum: "
        f"{toa_min}"
    )

    print(
        f"Slot width: "
        f"{slot_width}"
    )

    print(
        f"Number of slots: "
        f"{n_slots:,}"
    )

    print_memory_estimate(
        N_BANDS,
        n_slots
    )

    print(
        f"\nFinal grid shape: "
        f"{occupancy_grid.shape}"
    )

    print(
        f"Occupied cells: "
        f"{int(occupancy_grid.sum()):,}"
    )

    print(
        f"Total cells: "
        f"{occupancy_grid.size:,}"
    )

    return (
        occupancy_grid,
        amplitude_grid,
        pw_grid,
        aoa_grid,
        slot_width
    )