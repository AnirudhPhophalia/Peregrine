import numpy as np


def calculate_scan_metrics(
    occupancy_grid,
    actions,
    observations,
    truths,
    scanned
):

    actions = np.asarray(
        actions
    )

    observations = np.asarray(
        observations
    )

    truths = np.asarray(
        truths
    )

    scanned = np.asarray(
        scanned
    )

    n_slots = len(actions)

    global_active = (
        occupancy_grid[
            :,
            :n_slots
        ].sum(axis=0) > 0
    )

    total_active_slots = int(
        global_active.sum()
    )

    active_scans = (
        (truths == 1)
        & (scanned == 1)
    )

    idle_scans = (
        (truths == 0)
        & (scanned == 1)
    )

    detections = int(
        np.sum(
            active_scans
            & (observations == 1)
        )
    )

    false_alarms = int(
        np.sum(
            idle_scans
            & (observations == 1)
        )
    )

    active_observations = int(
        np.sum(active_scans)
    )

    idle_observations = int(
        np.sum(idle_scans)
    )

    receiver_pod = (
        detections
        / active_observations
        if active_observations > 0
        else 0.0
    )

    interception_rate = (
        detections
        / total_active_slots
        if total_active_slots > 0
        else 0.0
    )

    pofa = (
        false_alarms
        / idle_observations
        if idle_observations > 0
        else 0.0
    )

    avg_intercept_delay = (
        compute_intercept_delay(
            occupancy_grid,
            actions,
            observations,
            scanned
        )
    )

    return {
        "detections": detections,
        "false_alarms": false_alarms,
        "total_active_slots": (
            total_active_slots
        ),
        "active_observations": (
            active_observations
        ),
        "idle_observations": (
            idle_observations
        ),
        "receiver_POD": round(
            receiver_pod,
            4
        ),
        "interception_rate": round(
            interception_rate,
            4
        ),
        "POFA": round(
            pofa,
            4
        ),
        "avg_intercept_time_slots": (
            avg_intercept_delay
        )
    }


def calculate_stare_metrics(
    occupancy_grid,
    observations
):

    observations = np.asarray(
        observations
    )

    active = (
        occupancy_grid == 1
    )

    idle = (
        occupancy_grid == 0
    )

    detections = int(
        np.sum(
            active
            & (observations == 1)
        )
    )

    false_alarms = int(
        np.sum(
            idle
            & (observations == 1)
        )
    )

    active_cells = int(
        np.sum(active)
    )

    idle_cells = int(
        np.sum(idle)
    )

    pod = (
        detections
        / active_cells
        if active_cells > 0
        else 0.0
    )

    pofa = (
        false_alarms
        / idle_cells
        if idle_cells > 0
        else 0.0
    )

    return {
        "detections": detections,
        "false_alarms": false_alarms,
        "active_cells": active_cells,
        "idle_cells": idle_cells,
        "POD": round(
            pod,
            4
        ),
        "POFA": round(
            pofa,
            4
        )
    }


def compute_intercept_delay(
    occupancy_grid,
    actions,
    observations,
    scanned
):

    n_bands, n_slots = (
        occupancy_grid.shape
    )

    delays = []

    for band in range(
        n_bands
    ):

        active = (
            occupancy_grid[
                band,
                :n_slots
            ] == 1
        )

        start = None

        caught = False

        for t in range(
            n_slots
        ):

            if (
                active[t]
                and start is None
            ):

                start = t
                caught = False

            if (
                start is not None
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
                start is not None
                and not active[t]
            ):

                start = None
                caught = False

    if not delays:
        return None

    return round(
        float(
            np.mean(delays)
        ),
        2
    )


def print_scan_metrics(
    name,
    results
):

    print("\n")
    print("=" * 60)
    print(
        f"{name} RESULTS"
    )
    print("=" * 60)

    print(
        f"Detections: "
        f"{results['detections']}"
    )

    print(
        f"False alarms: "
        f"{results['false_alarms']}"
    )

    print(
        f"Total active slots: "
        f"{results['total_active_slots']}"
    )

    print(
        f"Active observations: "
        f"{results['active_observations']}"
    )

    print(
        f"Receiver POD: "
        f"{results['receiver_POD']:.4f}"
    )

    print(
        f"Interception rate: "
        f"{results['interception_rate']:.4f}"
    )

    print(
        f"POFA: "
        f"{results['POFA']:.4f}"
    )

    if (
        results[
            "avg_intercept_time_slots"
        ]
        is not None
    ):

        print(
            f"Avg intercept delay: "
            f"{results['avg_intercept_time_slots']} slots"
        )

    else:

        print(
            "Avg intercept delay: N/A"
        )

    print("=" * 60)


def print_stare_metrics(
    results
):

    print("\n")
    print("=" * 60)
    print("STARE RECEIVER RESULTS")
    print("=" * 60)

    print(
        f"Detections: "
        f"{results['detections']}"
    )

    print(
        f"False alarms: "
        f"{results['false_alarms']}"
    )

    print(
        f"Active cells: "
        f"{results['active_cells']}"
    )

    print(
        f"POD: "
        f"{results['POD']:.4f}"
    )

    print(
        f"POFA: "
        f"{results['POFA']:.4f}"
    )

    print("=" * 60)