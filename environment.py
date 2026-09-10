import numpy as np

from config import (
    NOISE_FLOOR_DB,
    NOISE_STD_DB,
    RECEIVER_SENSITIVITY_DB,
    DETECTION_SNR_50_DB,
    DETECTION_SLOPE,
    BASE_FALSE_ALARM_RATE,
    DWELL_SLOTS,
    RETUNE_SLOTS
)


class ScanEnvironment:

    def __init__(
        self,
        occupancy_grid,
        amplitude_grid,
        pw_grid,
        aoa_grid,
        seed=0
    ):

        self.occupancy_grid = occupancy_grid
        self.amplitude_grid = amplitude_grid
        self.pw_grid = pw_grid
        self.aoa_grid = aoa_grid

        self.n_bands, self.n_slots = (
            occupancy_grid.shape
        )

        self.rng = np.random.default_rng(seed)

        self.reset()

    # Reset
    def reset(self):

        self.t = 0

        self.current_band = None
        self.target_band = None

        self.retune_remaining = 0
        self.dwell_remaining = 0

        return self.t

    # Noise
    def sample_noise(self):

        return self.rng.normal(
            NOISE_FLOOR_DB,
            NOISE_STD_DB
        )

    # Detection probability
    def detection_probability(
        self,
        amplitude,
        noise
    ):

        snr = amplitude - noise

        # Below receiver sensitivity:
        # essentially impossible to detect.
        if amplitude < RECEIVER_SENSITIVITY_DB:
            return 0.0

        x = (
            DETECTION_SLOPE
            * (snr - DETECTION_SNR_50_DB)
        )

        # Numerically stable sigmoid
        x = np.clip(x, -50, 50)

        return 1.0 / (1.0 + np.exp(-x))

    # Observe current band
    def observe(self, band):

        truth = int(
            self.occupancy_grid[
                band,
                self.t
            ]
        )

        noise = self.sample_noise()

        amplitude = float(
            self.amplitude_grid[
                band,
                self.t
            ]
        )

        pw = float(
            self.pw_grid[
                band,
                self.t
            ]
        )

        aoa = float(
            self.aoa_grid[
                band,
                self.t
            ]
        )

        # Active signal
        if truth == 1:

            p_detect = self.detection_probability(
                amplitude,
                noise
            )

            obs = int(
                self.rng.random()
                < p_detect
            )

            snr = amplitude - noise

            return (
                obs,
                truth,
                {
                    "scanned": True,
                    "retuning": False,
                    "band": band,
                    "amplitude": amplitude,
                    "noise": noise,
                    "snr": snr,
                    "pw": pw,
                    "aoa": aoa,
                    "p_detect": p_detect
                }
            )

        # Empty spectrum
        obs = int(
            self.rng.random()
            < BASE_FALSE_ALARM_RATE
        )

        return (
            obs,
            truth,
            {
                "scanned": True,
                "retuning": False,
                "band": band,
                "amplitude": None,
                "noise": noise,
                "snr": None,
                "pw": 0.0,
                "aoa": np.nan,
                "p_detect": None
            }
        )

    # Step
    def step(self, requested_band):

        if not (
            0 <= requested_band < self.n_bands
        ):
            raise ValueError(
                "Invalid band index."
            )

        if self.t >= self.n_slots:
            raise RuntimeError(
                "Simulation finished. Call reset()."
            )

        # Currently retuning
        if self.retune_remaining > 0:

            actual_band = (
                self.target_band
            )

            self.retune_remaining -= 1

            obs = 0
            truth = 0

            info = {
                "scanned": False,
                "retuning": True,
                "band": actual_band,
                "amplitude": None,
                "noise": None,
                "snr": None,
                "pw": 0.0,
                "aoa": np.nan,
                "p_detect": None
            }

            done = (
                self.t == self.n_slots - 1
            )

            self.t += 1

            return obs, truth, done, info

        # During dwell
        if self.dwell_remaining > 0:

            actual_band = self.current_band

            obs, truth, info = self.observe(
                actual_band
            )

            info["dwell"] = True

            self.dwell_remaining -= 1

            done = (
                self.t == self.n_slots - 1
            )

            self.t += 1

            return obs, truth, done, info

        # New band requested
        if (
            self.current_band is None
            or requested_band != self.current_band
        ):

            self.target_band = requested_band

            self.retune_remaining = max(
                RETUNE_SLOTS - 1,
                0
            )

            self.current_band = requested_band

            # First slot is spent retuning.
            obs = 0
            truth = 0

            info = {
                "scanned": False,
                "retuning": True,
                "band": requested_band,
                "amplitude": None,
                "noise": None,
                "snr": None,
                "pw": 0.0,
                "aoa": np.nan,
                "p_detect": None
            }

            done = (
                self.t == self.n_slots - 1
            )

            self.t += 1

            return obs, truth, done, info

        # Scan
        obs, truth, info = self.observe(
            self.current_band
        )

        self.dwell_remaining = max(
            DWELL_SLOTS - 1,
            0
        )

        done = (
            self.t == self.n_slots - 1
        )

        self.t += 1

        return obs, truth, done, info