import numpy as np


def calculate_metrics(
    x_grid,
    actions,
    observations,
    truths,
    scanned
):

    actions = np.asarray(actions)
    observations = np.asarray(observations)
    truths = np.asarray(truths)
    scanned = np.asarray(scanned)

    n_slots = len(actions)

    global_active = (
        x_grid[:, :n_slots].sum(axis=0) > 0
    )

    total_active_slots = int(
        global_active.sum()
    )

    # Actual successful detections
    detections = int(
        np.sum(
            (truths == 1)
            & (observations == 1)
            & (scanned == 1)
        )
    )

    # False alarms
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

    # POD
    POD = (
        detections / total_active_slots
        if total_active_slots > 0
        else 0.0
    )

    # POFA
    POFA = (
        false_alarms / idle_scans
        if idle_scans > 0
        else 0.0
    )

    # Capture rate
    capture_rate = POD

    # Missed active slots
    missed_detections = max(
        total_active_slots - detections,
        0
    )

    # Spectrum coverage
    scanned_bands = set(
        actions[actions >= 0]
    )

    n_bands = x_grid.shape[0]

    coverage = (
        len(scanned_bands) / n_bands
        if n_bands > 0
        else 0.0
    )

    # Retuning / scan cost
    total_scans = int(
        scanned.sum()
    )

    total_time = len(actions)

    scan_fraction = (
        total_scans / total_time
        if total_time > 0
        else 0.0
    )

    # Detection delay
    delays = _compute_detection_delays(
        x_grid,
        actions,
        observations,
        scanned
    )

    if delays:

        mean_delay = float(
            np.mean(delays)
        )

        median_delay = float(
            np.median(delays)
        )

        p95_delay = float(
            np.percentile(delays, 95)
        )

    else:

        mean_delay = None
        median_delay = None
        p95_delay = None

    # Results
    return {

        "detections": detections,

        "missed_detections":
            missed_detections,

        "false_alarms":
            false_alarms,

        "total_active_events":
            total_active_slots,

        "POD":
            round(POD, 4),

        "POFA":
            round(POFA, 4),

        "capture_rate":
            round(capture_rate, 4),

        "mean_detection_delay":
            mean_delay,

        "median_detection_delay":
            median_delay,

        "p95_detection_delay":
            p95_delay,

        "spectrum_coverage":
            round(coverage, 4),

        "total_scans":
            total_scans,

        "scan_fraction":
            round(scan_fraction, 4)
    }


def _compute_detection_delays(
    x_grid,
    actions,
    observations,
    scanned
):

    n_bands, n_slots = x_grid.shape

    delays = []

    for band in range(n_bands):

        active = False
        start = None
        caught = False

        for t in range(
            min(n_slots, len(actions))
        ):

            is_active = (
                x_grid[band, t] == 1
            )

            # New pulse/burst
            if is_active and not active:

                active = True
                start = t
                caught = False

            # Detection during burst
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

            # Burst ended
            if active and not is_active:

                active = False
                start = None

    return delays


def print_metrics(name, results):

    print("\n" + "=" * 55)
    print(f"{name} RESULTS")
    print("=" * 55)

    print(
        f"Detections:          "
        f"{results['detections']}"
    )

    print(
        f"Missed detections:   "
        f"{results['missed_detections']}"
    )

    print(
        f"False alarms:        "
        f"{results['false_alarms']}"
    )

    print(
        f"Total active events: "
        f"{results['total_active_events']}"
    )

    print()

    print(
        f"POD:                 "
        f"{results['POD']:.4f}"
    )

    print(
        f"POFA:                "
        f"{results['POFA']:.4f}"
    )

    print(
        f"Capture rate:        "
        f"{results['capture_rate']:.2%}"
    )

    print(
        f"Spectrum coverage:   "
        f"{results['spectrum_coverage']:.2%}"
    )

    print(
        f"Total scans:         "
        f"{results['total_scans']}"
    )

    print(
        f"Scan fraction:       "
        f"{results['scan_fraction']:.2%}"
    )

    if results["mean_detection_delay"] is not None:

        print(
            f"Mean delay:          "
            f"{results['mean_detection_delay']:.2f} slots"
        )

        print(
            f"Median delay:        "
            f"{results['median_detection_delay']:.2f} slots"
        )

        print(
            f"95th percentile:     "
            f"{results['p95_detection_delay']:.2f} slots"
        )

    else:

        print(
            "Detection delay:     N/A"
        )

    print("=" * 55)