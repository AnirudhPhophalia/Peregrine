import numpy as np


class RoundRobinScheduler:
    def __init__(self, n_bands):
        self.n_bands = n_bands
        self.next_band = 0

    def select_action(self, t=None):
        # Called only when the receiver is ready.
        # Therefore one action corresponds to one frequency visit,
        # not one raw time slot.
        action = self.next_band

        self.next_band = (
            self.next_band + 1
        ) % self.n_bands

        return action

    def update(self, action, obs):
        pass


class UCBScheduler:
    def __init__(self, n_bands):
        self.n_bands = n_bands

        self.counts = np.zeros(
            n_bands,
            dtype=np.int64,
        )

        self.successes = np.zeros(
            n_bands,
            dtype=np.float64,
        )

    def select_action(self, t=None):
        # Force every band to be sampled once.
        for i in range(self.n_bands):
            if self.counts[i] == 0:
                return i

        total = max(
            int(np.sum(self.counts)),
            1,
        )

        p_hat = (
            self.successes
            / np.maximum(
                self.counts,
                1,
            )
        )

        bonus = np.sqrt(
            2.0
            * np.log(total + 1.0)
            / np.maximum(
                self.counts,
                1,
            )
        )

        return int(
            np.argmax(
                p_hat + bonus
            )
        )

    def update(self, action, obs):
        if action < 0:
            return

        self.counts[action] += 1
        self.successes[action] += obs


class RestlessBanditScheduler:
    def __init__(
        self,
        n_bands,
        prior=0.05,
        decay=0.98,
        exploration=0.2,
    ):
        self.n_bands = n_bands
        self.prior = prior
        self.decay = decay
        self.exploration = exploration

        self.beliefs = np.full(
            n_bands,
            prior,
            dtype=np.float64,
        )

        self.counts = np.zeros(
            n_bands,
            dtype=np.int64,
        )

    def select_action(self, t=None):
        step = (
            0
            if t is None
            else t
        )

        uncertainty = np.sqrt(
            np.log(step + 2.0)
            / (self.counts + 1.0)
        )

        scores = (
            self.beliefs
            + self.exploration
            * uncertainty
        )

        return int(
            np.argmax(scores)
        )

    def update(self, action, obs):
        self.beliefs = (
            self.decay
            * self.beliefs
            + (1.0 - self.decay)
            * self.prior
        )

        if action < 0:
            return

        self.counts[action] += 1

        alpha = 1.0 / np.sqrt(
            self.counts[action]
        )

        self.beliefs[action] = (
            (1.0 - alpha)
            * self.beliefs[action]
            + alpha * obs
        )


class POMDPScheduler:
    def __init__(
        self,
        n_bands,
        p_on=0.02,
        p_stay=0.90,
        exploration=0.1,
    ):
        self.n_bands = n_bands

        self.beliefs = np.full(
            n_bands,
            0.05,
            dtype=np.float64,
        )

        self.p_on = p_on
        self.p_stay = p_stay
        self.exploration = exploration

    def predict(self):
        self.beliefs = (
            self.beliefs
            * self.p_stay
            + (1.0 - self.beliefs)
            * self.p_on
        )

    def select_action(self, t=None):
        uncertainty = (
            4.0
            * self.beliefs
            * (1.0 - self.beliefs)
        )

        scores = (
            self.beliefs
            + self.exploration
            * uncertainty
        )

        return int(
            np.argmax(scores)
        )

    def update(
        self,
        action,
        obs,
        p_detect=0.8,
        p_false_alarm=0.01,
    ):
        if action < 0:
            return

        self.predict()

        prior = self.beliefs[action]

        if obs == 1:
            numerator = (
                p_detect * prior
            )

            denominator = (
                p_detect * prior
                + p_false_alarm
                * (1.0 - prior)
            )

        else:
            numerator = (
                (1.0 - p_detect)
                * prior
            )

            denominator = (
                (1.0 - p_detect)
                * prior
                + (1.0 - p_false_alarm)
                * (1.0 - prior)
            )

        if denominator > 0:
            self.beliefs[action] = (
                numerator / denominator
            )


def run_scheduler(
    occupancy_grid,
    amplitude_grid,
    pw_grid,
    aoa_grid,
    scheduler,
    seed=0,
):
    from environment import ScanEnvironment

    env = ScanEnvironment(
        occupancy_grid=occupancy_grid,
        amplitude_grid=amplitude_grid,
        pw_grid=pw_grid,
        aoa_grid=aoa_grid,
        seed=seed,
    )

    env.reset()

    n_slots = occupancy_grid.shape[1]

    actions = np.full(
        n_slots,
        -1,
        dtype=np.int32,
    )

    observations = np.zeros(
        n_slots,
        dtype=np.int8,
    )

    truths = np.zeros(
        n_slots,
        dtype=np.int8,
    )

    scanned = np.zeros(
        n_slots,
        dtype=np.int8,
    )

    snr = np.full(
        n_slots,
        np.nan,
        dtype=np.float32,
    )

    amplitudes = np.full(
        n_slots,
        np.nan,
        dtype=np.float32,
    )

    p_detect = np.full(
        n_slots,
        np.nan,
        dtype=np.float32,
    )

    p_false_alarm = np.full(
        n_slots,
        np.nan,
        dtype=np.float32,
    )

    done = False
    t = 0
    requested_action = None

    while not done:
        # CRITICAL FIX:
        # Do not choose a new band every raw time slot.
        # The receiver must finish retuning + dwell first.
        if requested_action is None:
            requested_action = scheduler.select_action(t)

        (
            obs,
            truth,
            done,
            info,
        ) = env.step(
            requested_action
        )

        if info["scanned"]:
            actions[t] = info["band"]
            scanned[t] = 1

            requested_action = None

            if isinstance(
                scheduler,
                POMDPScheduler,
            ):
                scheduler.update(
                    info["band"],
                    obs,
                    p_detect=(
                        info["p_detect"]
                        if info["p_detect"]
                        is not None
                        else 0.8
                    ),
                    p_false_alarm=(
                        info["p_false_alarm"]
                        if info["p_false_alarm"]
                        is not None
                        else 0.01
                    ),
                )
            else:
                scheduler.update(
                    info["band"],
                    obs,
                )

            if info["snr"] is not None:
                snr[t] = info["snr"]

            if (
                info["amplitude"]
                is not None
            ):
                amplitudes[t] = (
                    info["amplitude"]
                )

            if (
                info["p_detect"]
                is not None
            ):
                p_detect[t] = (
                    info["p_detect"]
                )

            if (
                info["p_false_alarm"]
                is not None
            ):
                p_false_alarm[t] = (
                    info["p_false_alarm"]
                )

        observations[t] = obs
        truths[t] = truth

        t += 1

    return (
        actions,
        observations,
        truths,
        scanned,
        snr,
        amplitudes,
        p_detect,
        p_false_alarm,
    )
