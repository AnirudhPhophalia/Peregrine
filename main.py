import time

from build_grid import build_grid
from dataset_loader import load_dataset
from inspect_dataset import inspect_dataset
from metrics import calculate_metrics, print_metrics
from scheduler import (
    POMDPScheduler,
    RestlessBanditScheduler,
    RoundRobinScheduler,
    UCBScheduler,
    run_scheduler,
)


def run_experiment(
    name,
    scheduler,
    occupancy_grid,
    amplitude_grid,
    pw_grid,
    aoa_grid,
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
        p_detect,
        p_false_alarm,
    ) = run_scheduler(
        occupancy_grid=occupancy_grid,
        amplitude_grid=amplitude_grid,
        pw_grid=pw_grid,
        aoa_grid=aoa_grid,
        scheduler=scheduler,
        seed=0,
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
        scanned,
    )

    print_metrics(
        name,
        results,
    )

    valid_pd = (
        p_detect[
            ~(
                p_detect != p_detect
            )
        ]
    )

    valid_pfa = (
        p_false_alarm[
            ~(
                p_false_alarm
                != p_false_alarm
            )
        ]
    )

    print(
        f"\nRuntime: {runtime:.4f} seconds"
    )

    if len(valid_pd):
        print(
            f"Mean dynamic p_detect: "
            f"{valid_pd.mean():.4f}"
        )

    if len(valid_pfa):
        print(
            f"Mean dynamic p_false_alarm: "
            f"{valid_pfa.mean():.4f}"
        )

    return results, runtime


def main():
    print("\nLOADING STARE DATASET...")
    print(
        "STARE data is used as the RF ground truth."
    )
    print(
        "The realistic scan receiver is simulated "
        "by this project."
    )

    df = load_dataset(
        mode="stare"
    )

    inspect_dataset(df)

    print(
        "\nBUILDING RF GROUND TRUTH..."
    )

    (
        occupancy_grid,
        amplitude_grid,
        pw_grid,
        aoa_grid,
        slot_width,
    ) = build_grid(
        df=df,
        freq_col=1,
        toa_col=0,
        pw_col=2,
        aoa_col=3,
        amp_col=4,
    )

    n_bands = (
        occupancy_grid.shape[0]
    )

    print(
        f"\nReceiver bands: {n_bands}"
    )

    print(
        f"Time slots: "
        f"{occupancy_grid.shape[1]:,}"
    )

    print(
        f"Slot width: "
        f"{slot_width:.3f} us"
    )

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
            ),
    }

    all_results = {}

    for (
        name,
        scheduler,
    ) in schedulers.items():

        results, runtime = (
            run_experiment(
                name,
                scheduler,
                occupancy_grid,
                amplitude_grid,
                pw_grid,
                aoa_grid,
            )
        )

        all_results[name] = {
            "metrics": results,
            "runtime": runtime,
        }

    print("\n")
    print(
        "=" * 100
    )
    print(
        "FINAL SCHEDULER COMPARISON"
    )
    print(
        "=" * 100
    )

    names = list(
        schedulers.keys()
    )

    print(
        f"{'Metric':<25}"
        + "".join(
            f"{name:>18}"
            for name in names
        )
    )

    print(
        "-" * 100
    )

    metric_rows = [
#        (
#            "Capture Rate",
#            "capture_rate",
#        ),
        (
            "POD",
            "POD",
        ),
        (
            "POFA",
            "POFA",
        ),
        (
            "Mean Delay",
            "mean_detection_delay",
        ),
        (
            "Spectrum Coverage",
            "spectrum_coverage",
        ),
        (
            "Detections",
            "detections",
        ),
        (
            "False Alarms",
            "false_alarms",
        ),
        (
            "Runtime (seconds)",
            None,
        ),
    ]

    for label, key in metric_rows:
        print(
            f"{label:<25}",
            end="",
        )

        for name in names:
            if key is None:
                value = (
                    all_results[name]
                    ["runtime"]
                )

                text = f"{value:.3f}"

            else:
                value = (
                    all_results[name]
                    ["metrics"]
                    [key]
                )

                if value is None:
                    text = "N/A"

                elif (
                    "rate" in key
                    or key in {
                        "POD",
                        "POFA",
                        "spectrum_coverage",
                    }
                ):
                    text = f"{value:.4f}"

                else:
                    text = str(value)

            print(
                f"{text:>18}",
                end="",
            )

        print()

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
