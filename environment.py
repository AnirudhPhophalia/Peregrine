import numpy as np

from config import (
    BASE_FALSE_ALARM_RATE,
    DETECTION_SLOPE,
    DETECTION_SNR_50_DB,
    DWELL_SLOTS,
    FALSE_ALARM_SLOPE,
    FALSE_ALARM_THRESHOLD_DBM,
    INTERFERENCE_PENALTY_DB,
    INTERFERENCE_PROB,
    MAX_FALSE_ALARM_RATE,
    NOISE_FLOOR_DBM,
    NOISE_STD_DB,
    RECEIVER_SENSITIVITY_DBM,
    REFERENCE_PW_US,
    RETUNE_SLOTS,
    SIGNAL_DROP_PROB,
)


class ScanEnvironment:
    def __init__(
        self,
        occupancy_grid,
        amplitude_grid,
        pw_grid,
        aoa_grid,
        seed=0,
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

    def reset(self):
        self.t = 0
        self.current_band = None
        self.target_band = None

        self.retune_remaining = 0
        self.dwell_remaining = 0

        return self.t

    @property
    def receiver_ready(self):
        return (
            self.retune_remaining == 0
            and self.dwell_remaining == 0
        )

    def sample_noise(self):
        return self.rng.normal(
            NOISE_FLOOR_DBM,
            NOISE_STD_DB,
        )

    @staticmethod
    def _sigmoid(x):
        x = np.clip(
            x,
            -50.0,
            50.0,
        )

        return 1.0 / (
            1.0 + np.exp(-x)
        )

    def detection_probability(
        self,
        amplitude,
        noise,
        pw,
    ):
        if (
            amplitude
            < RECEIVER_SENSITIVITY_DBM
        ):
            return 0.0

        # Longer pulses provide additional integration gain.
        pw_gain = 5.0 * np.log10(
            max(
                pw,
                0.1,
            )
            / REFERENCE_PW_US
        )

        snr = amplitude - noise
        effective_snr = snr + pw_gain

        return float(
            self._sigmoid(
                DETECTION_SLOPE
                * (
                    effective_snr
                    - DETECTION_SNR_50_DB
                )
            )
        )

    def false_alarm_probability(
        self,
        noise,
    ):
        # As the instantaneous noise floor rises above the
        # false-alarm threshold, false-alarm probability rises.
        pressure = (
            FALSE_ALARM_SLOPE
            * (
                noise
                - FALSE_ALARM_THRESHOLD_DBM
            )
        )

        stress = self._sigmoid(
            pressure
        )

        p = BASE_FALSE_ALARM_RATE * (
            0.25 + 3.75 * stress
        )

        return float(
            np.clip(
                p,
                0.0,
                MAX_FALSE_ALARM_RATE,
            )
        )

    def observe(self, band):
        truth = int(
            self.occupancy_grid[
                band,
                self.t,
            ]
        )

        noise = self.sample_noise()

        interference = (
            self.rng.random()
            < INTERFERENCE_PROB
        )

        effective_noise = (
            noise + INTERFERENCE_PENALTY_DB
            if interference
            else noise
        )

        amplitude = float(
            self.amplitude_grid[
                band,
                self.t,
            ]
        )

        pw = float(
            self.pw_grid[
                band,
                self.t,
            ]
        )

        aoa = float(
            self.aoa_grid[
                band,
                self.t,
            ]
        )

        p_false_alarm = (
            self.false_alarm_probability(
                effective_noise
            )
        )

        if truth == 1:
            p_detect = (
                self.detection_probability(
                    amplitude,
                    effective_noise,
                    pw,
                )
            )

            if (
                self.rng.random()
                < SIGNAL_DROP_PROB
            ):
                p_detect = 0.0

            obs = int(
                self.rng.random()
                < p_detect
            )

            snr = (
                amplitude
                - effective_noise
            )

            return (
                obs,
                truth,
                {
                    "scanned": True,
                    "retuning": False,
                    "dwell": False,
                    "band": band,
                    "amplitude": amplitude,
                    "noise": effective_noise,
                    "snr": snr,
                    "pw": pw,
                    "aoa": aoa,
                    "p_detect": p_detect,
                    "p_false_alarm": p_false_alarm,
                    "interference": interference,
                },
            )

        obs = int(
            self.rng.random()
            < p_false_alarm
        )

        return (
            obs,
            truth,
            {
                "scanned": True,
                "retuning": False,
                "dwell": False,
                "band": band,
                "amplitude": None,
                "noise": effective_noise,
                "snr": None,
                "pw": 0.0,
                "aoa": np.nan,
                "p_detect": None,
                "p_false_alarm": p_false_alarm,
                "interference": interference,
            },
        )

    def _retune_info(self, band):
        return {
            "scanned": False,
            "retuning": True,
            "dwell": False,
            "band": band,
            "amplitude": None,
            "noise": None,
            "snr": None,
            "pw": 0.0,
            "aoa": np.nan,
            "p_detect": None,
            "p_false_alarm": None,
            "interference": False,
        }

    def step(self, requested_band):
        if not (
            0 <= requested_band < self.n_bands
        ):
            raise ValueError(
                "Invalid band index."
            )

        if self.t >= self.n_slots:
            raise RuntimeError(
                "Simulation finished. "
                "Call reset()."
            )

        # Retuning consumes slots and produces no observation.
        if self.retune_remaining > 0:
            self.retune_remaining -= 1

            info = self._retune_info(
                self.target_band
            )

            done = (
                self.t
                == self.n_slots - 1
            )

            self.t += 1

            return (
                0,
                0,
                done,
                info,
            )

        # Once tuned, keep observing the same band during dwell.
        if self.dwell_remaining > 0:
            (
                obs,
                truth,
                info,
            ) = self.observe(
                self.current_band
            )

            info["dwell"] = True

            self.dwell_remaining -= 1

            done = (
                self.t
                == self.n_slots - 1
            )

            self.t += 1

            return (
                obs,
                truth,
                done,
                info,
            )

        # A different band requires retuning.
        if (
            self.current_band is None
            or requested_band
            != self.current_band
        ):
            self.target_band = (
                requested_band
            )

            self.current_band = (
                requested_band
            )

            self.retune_remaining = max(
                RETUNE_SLOTS - 1,
                0,
            )

            info = self._retune_info(
                requested_band
            )

            done = (
                self.t
                == self.n_slots - 1
            )

            self.t += 1

            return (
                0,
                0,
                done,
                info,
            )

        # Receiver is tuned and available:
        # this is the first useful observation in the dwell.
        (
            obs,
            truth,
            info,
        ) = self.observe(
            self.current_band
        )

        self.dwell_remaining = max(
            DWELL_SLOTS - 1,
            0,
        )

        done = (
            self.t
            == self.n_slots - 1
        )

        self.t += 1

        return (
            obs,
            truth,
            done,
            info,
        )
