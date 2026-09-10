import time

from dataset_loader import load_dataset
from inspect_dataset import inspect_dataset
from build_grid import (
    build_grid,
    estimate_slot_width
)

from scheduler import (
    RoundRobinScheduler,
    UCBScheduler,
    RestlessBanditScheduler,
    POMDPScheduler,
    run_scheduler
)

from metrics import (
    calculate_metrics,
    print_metrics
)

from config import TARGET_SLOTS


def run_experiment(
    name,
    scheduler,
    occupancy_grid,
    amplitude_grid,
    pw_grid,
    aoa_grid
):

    print("\n" + "=" * 60)
    print(f"RUNNING: {name}")
    print("=" * 60)

    start = time.perf_counter()

    (
        actions,
        observations,
        truths,
        scanned,
        snr,
        amplitudes,
        p_detect
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

    results = calculate_metrics(

        occupancy_grid,

        actions,

        observations,

        truths,

        scanned
    )

    print_metrics(
        name,
        results
    )

    print(
        f"\nRuntime: {runtime:.4f} seconds"
    )

    return results, runtime


def main():
    print("\nLOADING DATASET...")
    df = load_dataset()
    inspect_dataset(df)
    # DATASET COLUMNS
    TOA_COL = 0
    FREQ_COL = 1
    PW_COL = 2
    AOA_COL = 3
    AMP_COL = 4
    # SLOT WIDTH
    print(
        "\nCALCULATING SLOT WIDTH..."
    )
    slot_width = estimate_slot_width(
        df[TOA_COL].to_numpy(),
        TARGET_SLOTS
    )
    print(
        "Slot width:",
        slot_width
    )

    # BUILD RF ENVIRONMENT
    print("\nBUILDING RF GROUND TRUTH...")

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

        amp_col=AMP_COL,

        slot_width=slot_width
    )

    # NUMBER OF BANDS
    n_bands = (
        occupancy_grid.shape[0]
    )

    # CREATE SCHEDULERS
    schedulers = {

        "ROUND ROBIN":
            RoundRobinScheduler(
                n_bands
            ),

        "UCB BANDIT":
            UCBScheduler(
                n_bands
            ),

        "RESTLESS BANDIT":
            RestlessBanditScheduler(
                n_bands
            ),

        "POMDP":
            POMDPScheduler(
                n_bands
            )
    }

    # RUN
    all_results = {}

    for name, scheduler in schedulers.items():

        results, runtime = run_experiment(

            name,

            scheduler,

            occupancy_grid,

            amplitude_grid,

            pw_grid,

            aoa_grid
        )

        all_results[name] = {

            "metrics": results,

            "runtime": runtime
        }

    # FINAL COMPARISON
    print("\n")

    print("=" * 100)
    print("FINAL SCHEDULER COMPARISON")
    print("=" * 100)

    names = [

        "ROUND ROBIN",

        "UCB BANDIT",

        "RESTLESS BANDIT",

        "POMDP"
    ]

    print(
        f"{'Metric':<25}"
        f"{'Round Robin':>16}"
        f"{'UCB':>16}"
        f"{'Restless':>16}"
        f"{'POMDP':>16}"
    )

    print("-" * 100)

    # Capture rate
    print(
        f"{'Capture Rate':<25}",
        end=""
    )

    for name in names:

        value = (
            all_results[name]
            ["metrics"]
            ["capture_rate"]
        )

        print(
            f"{value:>15.2%}",
            end=" "
        )

    print()

    # POD
    print(
        f"{'POD':<25}",
        end=""
    )

    for name in names:

        value = (
            all_results[name]
            ["metrics"]
            ["POD"]
        )

        print(
            f"{value:>15.4f}",
            end=" "
        )

    print()

    # POFA
    print(
        f"{'POFA':<25}",
        end=""
    )

    for name in names:

        value = (
            all_results[name]
            ["metrics"]
            ["POFA"]
        )

        print(
            f"{value:>15.4f}",
            end=" "
        )

    print()

    # Mean delay
    print(
        f"{'Mean Delay':<25}",
        end=""
    )

    for name in names:
        value = (
            all_results[name]
            ["metrics"]
            ["mean_detection_delay"]
        )
        if value is None:
            print(
                f"{'N/A':>15}",
                end=" "
            )
        else:
            print(
                f"{value:>15.2f}",
                end=" "
            )
    print()

    # Spectrum coverage
    print(
        f"{'Spectrum Coverage':<25}",
        end=""
    )

    for name in names:
        value = (
            all_results[name]
            ["metrics"]
            ["spectrum_coverage"]
        )
        print(
            f"{value:>15.2%}",
            end=" "
        )
    print()

    # Detections
    print(
        f"{'Detections':<25}",
        end=""
    )

    for name in names:
        value = (
            all_results[name]
            ["metrics"]
            ["detections"]
        )
        print(
            f"{value:>15}",
            end=" "
        )
    print()

    # Runtime
    print(
        f"{'Runtime (seconds)':<25}",
        end=""
    )

    for name in names:
        value = (
            all_results[name]
            ["runtime"]
        )
        print(
            f"{value:>15.4f}",
            end=" "
        )
    print()
    print("=" * 100)


if __name__ == "__main__":
    main()