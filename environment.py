import numpy as np

from config import (
    NOISE_FLOOR_DB,
    NOISE_STD_DB,
    RECEIVER_SENSITIVITY_DB,
    DETECTION_SNR_50_DB,
    DETECTION_SLOPE,
    FALSE_ALARM_THRESHOLD_DB,
    FALSE_ALARM_SLOPE,
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

        self.occupancy_grid = (
            occupancy_grid
        )

        self.amplitude_grid = (
            amplitude_grid
        )

        self.pw_grid = (
            pw_grid
        )

        self.aoa_grid = (
            aoa_grid
        )

        self.n_bands, self.n_slots = (
            occupancy_grid.shape
        )

        self.rng = np.random.default_rng(
            seed
        )

        self.reset()


    # Reset

    def reset(self):

        self.t = 0

        self.current_band = None

        self.target_band = None

        self.retune_remaining = 0

        self.dwell_remaining = 0

        return self.t


    # Noise model

    def sample_noise(self):

        return float(
            self.rng.normal(
                NOISE_FLOOR_DB,
                NOISE_STD_DB
            )
        )


    # SNR

    def calculate_snr(
        self,
        signal_db,
        noise_db
    ):

        return float(
            signal_db - noise_db
        )


    # Probability of detection

    def detection_probability(
        self,
        signal_db,
        noise_db
    ):

        if not np.isfinite(signal_db):

            return 0.0

        if (
            signal_db
            < RECEIVER_SENSITIVITY_DB
        ):

            return 0.0

        snr_db = self.calculate_snr(
            signal_db,
            noise_db
        )

        x = (
            DETECTION_SLOPE
            * (
                snr_db
                - DETECTION_SNR_50_DB
            )
        )

        x = np.clip(
            x,
            -50.0,
            50.0
        )

        probability = (
            1.0
            / (
                1.0
                + np.exp(-x)
            )
        )

        return float(
            probability
        )


    # Probability of false alarm

    def false_alarm_probability(
        self,
        noise_db
    ):

        x = (
            noise_db
            - FALSE_ALARM_THRESHOLD_DB
        ) / FALSE_ALARM_SLOPE

        x = np.clip(
            x,
            -50.0,
            50.0
        )

        probability = (
            1.0
            / (
                1.0
                + np.exp(-x)
            )
        )

        return float(
            probability
        )


    # Observe one band

    def observe(self, band):

        truth = int(
            self.occupancy_grid[
                band,
                self.t
            ]
        )

        noise_db = self.sample_noise()

        amplitude_db = float(
            self.amplitude_grid[
                band,
                self.t
            ]
        )

        pw_us = float(
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

        if truth == 1:

            snr_db = (
                self.calculate_snr(
                    amplitude_db,
                    noise_db
                )
            )

            p_detect = (
                self.detection_probability(
                    amplitude_db,
                    noise_db
                )
            )

            p_false_alarm = 0.0

            observation = int(
                self.rng.random()
                < p_detect
            )

        else:

            snr_db = None

            p_detect = 0.0

            p_false_alarm = (
                self.false_alarm_probability(
                    noise_db
                )
            )

            observation = int(
                self.rng.random()
                < p_false_alarm
            )

            amplitude_db = np.nan

            pw_us = 0.0

            aoa = np.nan

        return (
            observation,
            truth,
            {
                "scanned": True,
                "retuning": False,
                "dwell": False,
                "band": band,
                "amplitude": amplitude_db,
                "noise": noise_db,
                "snr": snr_db,
                "pw": pw_us,
                "aoa": aoa,
                "p_detect": p_detect,
                "p_false_alarm": (
                    p_false_alarm
                )
            }
        )


    # Scan receiver

    def scan_step(
        self,
        requested_band
    ):

        if not (
            0 <= requested_band
            < self.n_bands
        ):

            raise ValueError(
                "Invalid band index."
            )

        if self.t >= self.n_slots:

            raise RuntimeError(
                "Simulation finished. "
                "Call reset()."
            )

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
                0
            )

            self.dwell_remaining = 0

            info = {
                "scanned": False,
                "retuning": True,
                "dwell": False,
                "band": requested_band,
                "amplitude": np.nan,
                "noise": np.nan,
                "snr": None,
                "pw": 0.0,
                "aoa": np.nan,
                "p_detect": 0.0,
                "p_false_alarm": 0.0
            }

            done = (
                self.t
                >= self.n_slots - 1
            )

            self.t += 1

            return (
                0,
                0,
                done,
                info
            )

        if self.retune_remaining > 0:

            self.retune_remaining -= 1

            info = {
                "scanned": False,
                "retuning": True,
                "dwell": False,
                "band": self.current_band,
                "amplitude": np.nan,
                "noise": np.nan,
                "snr": None,
                "pw": 0.0,
                "aoa": np.nan,
                "p_detect": 0.0,
                "p_false_alarm": 0.0
            }

            done = (
                self.t
                >= self.n_slots - 1
            )

            self.t += 1

            return (
                0,
                0,
                done,
                info
            )

        observation, truth, info = (
            self.observe(
                self.current_band
            )
        )

        info["dwell"] = (
            self.dwell_remaining > 0
        )

        if self.dwell_remaining > 0:

            self.dwell_remaining -= 1

        else:

            self.dwell_remaining = max(
                DWELL_SLOTS - 1,
                0
            )

        done = (
            self.t
            >= self.n_slots - 1
        )

        self.t += 1

        return (
            observation,
            truth,
            done,
            info
        )


    # Stare receiver

    def stare_step(self):

        if self.t >= self.n_slots:

            raise RuntimeError(
                "Simulation finished. "
                "Call reset()."
            )

        observations = np.zeros(
            self.n_bands,
            dtype=np.int8
        )

        truths = np.zeros(
            self.n_bands,
            dtype=np.int8
        )

        amplitudes = np.full(
            self.n_bands,
            np.nan,
            dtype=np.float32
        )

        noises = np.full(
            self.n_bands,
            np.nan,
            dtype=np.float32
        )

        snrs = np.full(
            self.n_bands,
            np.nan,
            dtype=np.float32
        )

        p_detects = np.zeros(
            self.n_bands,
            dtype=np.float32
        )

        p_false_alarms = np.zeros(
            self.n_bands,
            dtype=np.float32
        )

        for band in range(
            self.n_bands
        ):

            (
                observation,
                truth,
                info
            ) = self.observe(
                band
            )

            observations[band] = (
                observation
            )

            truths[band] = truth

            amplitudes[band] = (
                info["amplitude"]
            )

            noises[band] = (
                info["noise"]
            )

            if info["snr"] is not None:

                snrs[band] = (
                    info["snr"]
                )

            p_detects[band] = (
                info["p_detect"]
            )

            p_false_alarms[band] = (
                info["p_false_alarm"]
            )

        done = (
            self.t
            >= self.n_slots - 1
        )

        self.t += 1

        return (
            observations,
            truths,
            amplitudes,
            noises,
            snrs,
            p_detects,
            p_false_alarms,
            done
        )


    # Generic step

    def step(
        self,
        requested_band,
        mode="scan"
    ):

        if mode == "scan":

            return self.scan_step(
                requested_band
            )

        if mode == "stare":

            raise ValueError(
                "Use stare_step() "
                "for stare mode."
            )

        raise ValueError(
            "mode must be "
            "'scan' or 'stare'."
        )