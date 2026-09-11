import numpy as np

# ROUND ROBIN
class RoundRobinScheduler:

    def __init__(self, n_bands):
        self.n_bands = n_bands

    def select_action(self, t):
        return t % self.n_bands

    def update(self, action, obs):
        pass

# UCB
class UCBScheduler:

    def __init__(self, n_bands):

        self.n_bands = n_bands

        self.counts = np.zeros(
            n_bands,
            dtype=np.int64
        )

        self.successes = np.zeros(
            n_bands,
            dtype=np.float64
        )

    def select_action(self, t):

        for i in range(self.n_bands):

            if self.counts[i] == 0:
                return i

        total = np.sum(self.counts)

        p_hat = (
            self.successes
            / self.counts
        )

        bonus = np.sqrt(
            2 * np.log(total + 1)
            / self.counts
        )

        scores = p_hat + bonus

        return int(np.argmax(scores))

    def update(self, action, obs):

        if action < 0:
            return

        self.counts[action] += 1
        self.successes[action] += obs


# RESTLESS BANDIT
class RestlessBanditScheduler:

    def __init__(
        self,
        n_bands,
        prior=0.05,
        decay=0.98,
        exploration=0.2
    ):

        self.n_bands = n_bands

        self.prior = prior
        self.decay = decay
        self.exploration = exploration

        self.beliefs = np.full(
            n_bands,
            prior,
            dtype=np.float64
        )

        self.counts = np.zeros(
            n_bands,
            dtype=np.int64
        )

    def select_action(self, t):

        uncertainty = np.sqrt(
            np.log(t + 2)
            / (self.counts + 1)
        )

        scores = (
            self.beliefs
            + self.exploration * uncertainty
        )

        return int(
            np.argmax(scores)
        )

    def update(self, action, obs):

        # Every band evolves.
        self.beliefs = (
            self.decay * self.beliefs
            +
            (1 - self.decay) * self.prior
        )

        if action < 0:
            return

        self.counts[action] += 1

        alpha = 1.0 / np.sqrt(
            self.counts[action]
        )

        self.beliefs[action] = (
            (1 - alpha)
            * self.beliefs[action]
            +
            alpha * obs
        )

# POMDP
class POMDPScheduler:

    def __init__(
        self,
        n_bands,
        p_on=0.02,
        p_stay=0.90,
        p_detect=0.8,
        p_false_alarm=0.1,
        exploration=0.1
    ):

        self.n_bands = n_bands

        self.beliefs = np.full(
            n_bands,
            0.05,
            dtype=np.float64
        )

        self.p_on = p_on
        self.p_stay = p_stay

        self.p_detect = p_detect
        self.p_false_alarm = p_false_alarm

        self.exploration = exploration

    def predict(self):

        self.beliefs = (
            self.beliefs * self.p_stay
            +
            (1 - self.beliefs) * self.p_on
        )

    def select_action(self, t):

        uncertainty = (
            4
            * self.beliefs
            * (1 - self.beliefs)
        )

        scores = (
            self.beliefs
            + self.exploration * uncertainty
        )

        return int(
            np.argmax(scores)
        )

    def update(self, action, obs):

        if action < 0:
            return

        self.predict()

        prior = self.beliefs[action]

        if obs == 1:

            numerator = (
                self.p_detect * prior
            )

            denominator = (
                self.p_detect * prior
                +
                self.p_false_alarm
                * (1 - prior)
            )

        else:

            numerator = (
                (1 - self.p_detect)
                * prior
            )

            denominator = (
                (1 - self.p_detect)
                * prior
                +
                (1 - self.p_false_alarm)
                * (1 - prior)
            )

        if denominator > 0:

            self.beliefs[action] = (
                numerator / denominator
            )

# RUN SCHEDULER
def run_scheduler(
    occupancy_grid,
    amplitude_grid,
    pw_grid,
    aoa_grid,
    scheduler,
    seed=0
):

    from environment import ScanEnvironment

    env = ScanEnvironment(
        occupancy_grid=occupancy_grid,
        amplitude_grid=amplitude_grid,
        pw_grid=pw_grid,
        aoa_grid=aoa_grid,
        seed=seed
    )

    env.reset()

    n_slots = occupancy_grid.shape[1]

    # -1 means the receiver was not scanning
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

    snr = np.full(
        n_slots,
        np.nan,
        dtype=np.float32
    )

    amplitudes = np.full(
        n_slots,
        np.nan,
        dtype=np.float32
    )

    p_detect = np.full(
        n_slots,
        np.nan,
        dtype=np.float32
    )

    done = False
    t = 0

    while not done:

        requested_action = (
            scheduler.select_action(t)
        )

        obs, truth, done, info = env.step(
            requested_action
        )

        actual_band = info["band"]

        if info["scanned"]:

            actions[t] = actual_band
            scanned[t] = 1

            scheduler.update(
                actual_band,
                obs
            )

            if info["snr"] is not None:
                snr[t] = info["snr"]

            if info["amplitude"] is not None:
                amplitudes[t] = (
                    info["amplitude"]
                )

            if info["p_detect"] is not None:
                p_detect[t] = (
                    info["p_detect"]
                )

        else:

            # No useful observation was obtained.
            scheduler.update(
                -1,
                0
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
        p_detect
    )