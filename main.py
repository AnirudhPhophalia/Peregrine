import time

from dataset_loader import load_dataset
from inspect_dataset import inspect_dataset
from build_grid import build_grid

from scheduler import (
    RoundRobinScheduler,
    UCBScheduler,
    run_scheduler,
    run_stare
)

from metrics import (
    calculate_scan_metrics,
    calculate_stare_metrics,
    print_scan_metrics,
    print_stare_metrics
)


def run_scan_experiment(
    name,
    scheduler,
    occupancy_grid,
    amplitude_grid,
    pw_grid,
    aoa_grid
):

    print("\n")
    print("=" * 60)
    print(
        f"RUNNING SCAN: {name}"
    )
    print("=" * 60)

    start = time.perf_counter()

    (
        actions,
        observations,
        truths,
        scanned,
        retuning,
        dwell,
        amplitudes,
        noises,
        snrs,
        p_detects,
        p_false_alarms,
        pws,
        aoas
    ) = run_scheduler(
        occupancy_grid=occupancy_grid,
        amplitude_grid=amplitude_grid,
        pw_grid=pw_grid,
        aoa_grid=aoa_grid,
        scheduler=scheduler,
        seed=0
    )

    runtime = (
        time.perf_counter()
        - start
    )

    results = calculate_scan_metrics(
        occupancy_grid=occupancy_grid,
        actions=actions,
        observations=observations,
        truths=truths,
        scanned=scanned
    )

    print_scan_metrics(
        name,
        results
    )

    print(
        f"\nRuntime: "
        f"{runtime:.4f} seconds"
    )

    print(
        f"Scanned slots: "
        f"{int(scanned.sum()):,}"
    )

    print(
        f"Retuning slots: "
        f"{int(retuning.sum()):,}"
    )

    print(
        f"Dwell slots: "
        f"{int(dwell.sum()):,}"
    )

    return (
        results,
        runtime
    )


def run_stare_experiment(
    occupancy_grid,
    amplitude_grid,
    pw_grid,
    aoa_grid
):

    print("\n")
    print("=" * 60)
    print("RUNNING STARE REFERENCE")
    print("=" * 60)

    start = time.perf_counter()

    (
        observations,
        truths,
        amplitudes,
        noises,
        snrs,
        p_detects,
        p_false_alarms
    ) = run_stare(
        occupancy_grid=occupancy_grid,
        amplitude_grid=amplitude_grid,
        pw_grid=pw_grid,
        aoa_grid=aoa_grid,
        seed=0
    )

    runtime = (
        time.perf_counter()
        - start
    )

    results = calculate_stare_metrics(
        occupancy_grid,
        observations
    )

    print_stare_metrics(
        results
    )

    print(
        f"\nRuntime: "
        f"{runtime:.4f} seconds"
    )

    return (
        results,
        runtime
    )


def main():

    print(
        "\nLOADING DATASET..."
    )

    df = load_dataset()

    inspect_dataset(df)

    print(
        "\nBUILDING RF "
        "GROUND TRUTH..."
    )

    TOA_COL = 0
    FREQ_COL = 1
    PW_COL = 2
    AOA_COL = 3
    AMP_COL = 4

    (
        occupancy_grid,
        amplitude_grid,
        pw_grid,
        aoa_grid,
        slot_width
    ) = build_grid(
        df=df,
        freq_col=FREQ_COL,
        toa_col=TOA_COL,
        pw_col=PW_COL,
        aoa_col=AOA_COL,
        amp_col=AMP_COL
    )

    print(
        f"\nUsing slot width: "
        f"{slot_width}"
    )

    n_bands = (
        occupancy_grid.shape[0]
    )

    print("\n")
    print("=" * 60)
    print("STARE REFERENCE")
    print("=" * 60)

    stare_results, stare_runtime = (
        run_stare_experiment(
            occupancy_grid,
            amplitude_grid,
            pw_grid,
            aoa_grid
        )
    )

    schedulers = {
        "ROUND ROBIN":
            RoundRobinScheduler(
                n_bands
            ),

        "UCB BANDIT":
            UCBScheduler(
                n_bands
            )
    }

    all_results = {}

    for name, scheduler in (
        schedulers.items()
    ):

        results, runtime = (
            run_scan_experiment(
                name,
                scheduler,
                occupancy_grid,
                amplitude_grid,
                pw_grid,
                aoa_grid
            )
        )

        all_results[name] = {
            "metrics": results,
            "runtime": runtime
        }

    print("\n")
    print("=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(
        f"{'Metric':<30}"
        f"{'STARE':>16}"
        f"{'ROUND ROBIN':>18}"
        f"{'UCB':>16}"
    )

    print("-" * 80)

    rr = all_results[
        "ROUND ROBIN"
    ]["metrics"]

    ucb = all_results[
        "UCB BANDIT"
    ]["metrics"]

    print(
        f"{'POD':<30}"
        f"{stare_results['POD']:>16.4f}"
        f"{rr['receiver_POD']:>18.4f}"
        f"{ucb['receiver_POD']:>16.4f}"
    )

    print(
        f"{'POFA':<30}"
        f"{stare_results['POFA']:>16.4f}"
        f"{rr['POFA']:>18.4f}"
        f"{ucb['POFA']:>16.4f}"
    )

    print(
        f"{'Interception Rate':<30}"
        f"{'N/A':>16}"
        f"{rr['interception_rate']:>18.4f}"
        f"{ucb['interception_rate']:>16.4f}"
    )

    print(
        f"{'Detections':<30}"
        f"{stare_results['detections']:>16}"
        f"{rr['detections']:>18}"
        f"{ucb['detections']:>16}"
    )

    print(
        f"{'False Alarms':<30}"
        f"{stare_results['false_alarms']:>16}"
        f"{rr['false_alarms']:>18}"
        f"{ucb['false_alarms']:>16}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()