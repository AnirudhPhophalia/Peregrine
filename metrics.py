import numpy as np


def calculate_metrics(
    x_grid,
    actions,
    observations,
    truths,
    scanned,
):
    actions = np.asarray(actions)
    observations = np.asarray(observations)
    truths = np.asarray(truths)
    scanned = np.asarray(scanned)

    n_slots = len(actions)

    # Count active band-time cells, not merely time slots in which
    # at least one signal exists. A scanner can miss a band while
    # another band is active in the same time slot.
    active_cells = (
        x_grid[:, :n_slots] == 1
    )

    total_active_events = int(
        active_cells.sum()
    )

    detections = int(
        np.sum(
            (truths == 1)
            & (observations == 1)
            & (scanned == 1)
        )
    )

    false_alarms = int(
        np.sum(
            (truths == 0)
            & (observations == 1)
            & (scanned == 1)
        )
    )

    idle_scans = int(
        np.sum(
            (truths == 0)
            & (scanned == 1)
        )
    )

    POD = (
        detections / total_active_events
        if total_active_events > 0
        else 0.0
    )

    POFA = (
        false_alarms / idle_scans
        if idle_scans > 0
        else 0.0
    )

    scanned_bands = set(
        actions[
            actions >= 0
        ]
    )

    n_bands = x_grid.shape[0]

    coverage = (
        len(scanned_bands) / n_bands
        if n_bands > 0
        else 0.0
    )

    total_scans = int(
        scanned.sum()
    )

    total_time = len(actions)

    scan_fraction = (
        total_scans / total_time
        if total_time > 0
        else 0.0
    )

    delays = _compute_detection_delays(
        x_grid,
        actions,
        observations,
        scanned,
    )

    if delays:
        mean_delay = float(
            np.mean(delays)
        )

        median_delay = float(
            np.median(delays)
        )

        p95_delay = float(
            np.percentile(
                delays,
                95,
            )
        )
    else:
        mean_delay = None
        median_delay = None
        p95_delay = None

    return {
        "detections": detections,
        "missed_detections": max(
            total_active_events
            - detections,
            0,
        ),
        "false_alarms": false_alarms,
        "total_active_events": total_active_events,
        "POD": round(POD, 4),
        "POFA": round(POFA, 4),
        "capture_rate": round(POD, 4),
        "mean_detection_delay": mean_delay,
        "median_detection_delay": median_delay,
        "p95_detection_delay": p95_delay,
        "spectrum_coverage": round(
            coverage,
            4,
        ),
        "total_scans": total_scans,
        "scan_fraction": round(
            scan_fraction,
            4,
        ),
    }


def _compute_detection_delays(
    x_grid,
    actions,
    observations,
    scanned,
):
    n_bands, n_slots = (
        x_grid.shape
    )

    delays = []

    for band in range(n_bands):
        active = False
        start = None
        caught = False

        for t in range(
            min(
                n_slots,
                len(actions),
            )
        ):
            is_active = (
                x_grid[band, t] == 1
            )

            if (
                is_active
                and not active
            ):
                active = True
                start = t
                caught = False

            if (
                active
                and not caught
                and scanned[t] == 1
                and actions[t] == band
                and observations[t] == 1
            ):
                delays.append(
                    t - start
                )
                caught = True

            if (
                active
                and not is_active
            ):
                active = False
                start = None

    return delays


def print_metrics(
    name,
    results,
):
    print("\n" + "=" * 55)
    print(f"{name} RESULTS")
    print("=" * 55)

    for key, value in results.items():
        print(
            f"{key:25}: {value}"
        )
