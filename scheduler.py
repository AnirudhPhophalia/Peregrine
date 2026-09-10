import numpy as np


# Round Robin

class RoundRobinScheduler:

    def __init__(
        self,
        n_bands
    ):

        self.n_bands = n_bands

        self.current_band = 0


    def select_action(
        self,
        t=None
    ):

        action = (
            self.current_band
        )

        self.current_band = (
            self.current_band + 1
        ) % self.n_bands

        return action


    def update(
        self,
        action,
        obs
    ):

        pass


# UCB

class UCBScheduler:

    def __init__(
        self,
        n_bands
    ):

        self.n_bands = n_bands

        self.counts = np.zeros(
            n_bands,
            dtype=np.int64
        )

        self.successes = np.zeros(
            n_bands,
            dtype=np.float64
        )


    def select_action(
        self,
        t=None
    ):

        for band in range(
            self.n_bands
        ):

            if (
                self.counts[band]
                == 0
            ):

                return band

        total = np.sum(
            self.counts
        )

        means = (
            self.successes
            / self.counts
        )

        bonus = np.sqrt(
            2.0
            * np.log(total + 1.0)
            / self.counts
        )

        scores = (
            means + bonus
        )

        return int(
            np.argmax(scores)
        )


    def update(
        self,
        action,
        obs
    ):

        if action < 0:
            return

        self.counts[action] += 1

        self.successes[action] += (
            obs
        )


# Scan experiment

def run_scheduler(
    occupancy_grid,
    amplitude_grid,
    pw_grid,
    aoa_grid,
    scheduler,
    seed=0
):

    from environment import (
        ScanEnvironment
    )

    env = ScanEnvironment(
        occupancy_grid=occupancy_grid,
        amplitude_grid=amplitude_grid,
        pw_grid=pw_grid,
        aoa_grid=aoa_grid,
        seed=seed
    )

    env.reset()

    n_slots = (
        occupancy_grid.shape[1]
    )

    actions = np.full(
        n_slots,
        -1,
        dtype=np.int32
    )

    observations = np.zeros(
        n_slots,
        dtype=np.int8
    )

    truths = np.zeros(
        n_slots,
        dtype=np.int8
    )

    scanned = np.zeros(
        n_slots,
        dtype=np.int8
    )

    retuning = np.zeros(
        n_slots,
        dtype=np.int8
    )

    dwell = np.zeros(
        n_slots,
        dtype=np.int8
    )

    amplitudes = np.full(
        n_slots,
        np.nan,
        dtype=np.float32
    )

    noises = np.full(
        n_slots,
        np.nan,
        dtype=np.float32
    )

    snrs = np.full(
        n_slots,
        np.nan,
        dtype=np.float32
    )

    p_detects = np.full(
        n_slots,
        np.nan,
        dtype=np.float32
    )

    p_false_alarms = np.full(
        n_slots,
        np.nan,
        dtype=np.float32
    )

    pws = np.full(
        n_slots,
        np.nan,
        dtype=np.float32
    )

    aoas = np.full(
        n_slots,
        np.nan,
        dtype=np.float32
    )

    done = False

    t = 0

    while not done:

        requested_band = (
            scheduler.select_action(t)
        )

        (
            observation,
            truth,
            done,
            info
        ) = env.step(
            requested_band,
            mode="scan"
        )

        observations[t] = (
            observation
        )

        truths[t] = truth

        retuning[t] = int(
            info["retuning"]
        )

        dwell[t] = int(
            info["dwell"]
        )

        if info["scanned"]:

            scanned[t] = 1

            actions[t] = (
                info["band"]
            )

            scheduler.update(
                info["band"],
                observation
            )

            amplitudes[t] = (
                info["amplitude"]
            )

            noises[t] = (
                info["noise"]
            )

            if info["snr"] is not None:

                snrs[t] = (
                    info["snr"]
                )

            p_detects[t] = (
                info["p_detect"]
            )

            p_false_alarms[t] = (
                info["p_false_alarm"]
            )

            pws[t] = (
                info["pw"]
            )

            aoas[t] = (
                info["aoa"]
            )

        else:

            scheduler.update(
                -1,
                0
            )

        t += 1

    return (
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
    )


# Stare experiment

def run_stare(
    occupancy_grid,
    amplitude_grid,
    pw_grid,
    aoa_grid,
    seed=0
):

    from environment import (
        ScanEnvironment
    )

    env = ScanEnvironment(
        occupancy_grid=occupancy_grid,
        amplitude_grid=amplitude_grid,
        pw_grid=pw_grid,
        aoa_grid=aoa_grid,
        seed=seed
    )

    env.reset()

    n_bands, n_slots = (
        occupancy_grid.shape
    )

    observations = np.zeros(
        (
            n_bands,
            n_slots
        ),
        dtype=np.int8
    )

    truths = np.zeros(
        (
            n_bands,
            n_slots
        ),
        dtype=np.int8
    )

    amplitudes = np.full(
        (
            n_bands,
            n_slots
        ),
        np.nan,
        dtype=np.float32
    )

    noises = np.full(
        (
            n_bands,
            n_slots
        ),
        np.nan,
        dtype=np.float32
    )

    snrs = np.full(
        (
            n_bands,
            n_slots
        ),
        np.nan,
        dtype=np.float32
    )

    p_detects = np.full(
        (
            n_bands,
            n_slots
        ),
        np.nan,
        dtype=np.float32
    )

    p_false_alarms = np.full(
        (
            n_bands,
            n_slots
        ),
        np.nan,
        dtype=np.float32
    )

    done = False

    while not done:

        (
            obs,
            truth,
            amplitude,
            noise,
            snr,
            p_detect,
            p_false_alarm,
            done
        ) = env.stare_step()

        t = env.t - 1

        observations[
            :,
            t
        ] = obs

        truths[
            :,
            t
        ] = truth

        amplitudes[
            :,
            t
        ] = amplitude

        noises[
            :,
            t
        ] = noise

        snrs[
            :,
            t
        ] = snr

        p_detects[
            :,
            t
        ] = p_detect

        p_false_alarms[
            :,
            t
        ] = p_false_alarm

    return (
        observations,
        truths,
        amplitudes,
        noises,
        snrs,
        p_detects,
        p_false_alarms
    )