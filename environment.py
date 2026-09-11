import numpy as np

from config import (
    FREQ_MIN_GHZ,
    FREQ_MAX_GHZ,
    RECEIVER_BANDWIDTH_BANDS,
    NOISE_FLOOR_DBM,
    NOISE_STD_DB,
    RECEIVER_SENSITIVITY_DBM,
    DETECTION_SNR_50_DB,
    DETECTION_SLOPE,
    BASE_FALSE_ALARM_RATE,
    DWELL_SLOTS,
    RETUNE_SLOTS,
    SIMULATION_DURATION_S,
    SLOT_DURATION_US,
    FALSE_ALARM_THRESHOLD_DB,
    FALSE_ALARM_SLOPE
)


class ScanEnvironment:

    # INITIALIZATION
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

        (
            self.n_bands,
            self.n_slots
        ) = occupancy_grid.shape

        self.rng = np.random.default_rng(
            seed
        )

        self.reset()

    # RESET
    def reset(self):

        self.t = 0

        self.current_band = None

        self.target_band = None

        self.retune_remaining = 0

        self.dwell_remaining = 0

        self.mode = "SCAN"

        return self.t

    # NOISE MODEL
    def sample_noise_power(self):

        return self.rng.normal(
            NOISE_FLOOR_DBM,
            NOISE_STD_DB
        )

    # SNR
    def calculate_snr(
        self,
        signal_power_dbm,
        noise_power_dbm
    ):

        return (
            signal_power_dbm
            - noise_power_dbm
        )

    # DETECTION PROBABILITY
    def detection_probability(
        self,
        signal_power_dbm,
        noise_power_dbm
    ):

        # Signal below receiver sensitivity cannot be
        # meaningfully detected.

        if (
            signal_power_dbm
            < RECEIVER_SENSITIVITY_DBM
        ):

            return 0.0

        snr_db = self.calculate_snr(signal_power_dbm, noise_power_dbm)

        x = (
            DETECTION_SLOPE
            * (snr_db - DETECTION_SNR_50_DB)
        )

        x = np.clip(
            x,
            -50,
            50
        )

        probability = (1.0/(1.0+np.exp(-x)))

        return probability

    # SCAN OBSERVATION
    def observe_band(
        self,
        band
    ):

        truth = int(
            self.occupancy_grid[
                band,
                self.t
            ]
        )

        noise_power = (
            self.sample_noise_power()
        )

        signal_power = float(
            self.amplitude_grid[
                band,
                self.t
            ]
        )

        pulse_width = float(
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

        # REAL SIGNAL
        if truth == 1:

            snr_db = self.calculate_snr(
                signal_power,
                noise_power
            )

            p_detect = (
                self.detection_probability(
                    signal_power,
                    noise_power
                )
            )

            detected = int(
                self.rng.random()
                < p_detect
            )

            return (
                detected,
                truth,
                {
                    "mode": "SCAN",
                    "scanned": True,
                    "retuning": False,

                    "band": band,

                    "signal_power_dbm":
                        signal_power,

                    "noise_power_dbm":
                        noise_power,

                    "snr_db":
                        snr_db,

                    "p_detect":
                        p_detect,

                    "pw_us":
                        pulse_width,

                    "aoa":
                        aoa
                }
            )

        # NO SIGNAL
        false_alarm = int(
            self.rng.random()
            < BASE_FALSE_ALARM_RATE
        )

        return (
            false_alarm,
            truth,
            {
                "mode": "SCAN",
                "scanned": True,
                "retuning": False,

                "band": band,

                "signal_power_dbm":
                    None,

                "noise_power_dbm":
                    noise_power,

                "snr_db":
                    None,

                "p_detect":
                    None,

                "pw_us":
                    0.0,

                "aoa":
                    np.nan
            }
        )

    # STARE / ORACLE OBSERVATION
    def stare(self):

        if self.t >= self.n_slots:

            raise RuntimeError(
                "Simulation has finished."
            )

        observations = np.zeros(
            self.n_bands,
            dtype=np.int8
        )

        truths = self.occupancy_grid[
            :,
            self.t
        ].copy()

        noise = np.zeros(
            self.n_bands,
            dtype=np.float32
        )

        snr = np.full(
            self.n_bands,
            np.nan,
            dtype=np.float32
        )

        p_detect = np.full(
            self.n_bands,
            np.nan,
            dtype=np.float32
        )

        # Oracle receiver sees every band simultaneously.
        for band in range(
            self.n_bands
        ):

            noise_power = (
                self.sample_noise_power()
            )

            noise[band] = (
                noise_power
            )

            if truths[band] == 1:

                signal_power = float(
                    self.amplitude_grid[
                        band,
                        self.t
                    ]
                )

                snr_db = (
                    self.calculate_snr(
                        signal_power,
                        noise_power
                    )
                )

                probability = (
                    self.detection_probability(
                        signal_power,
                        noise_power
                    )
                )

                detection = int(
                    self.rng.random()
                    < probability
                )

                observations[band] = (
                    detection
                )

                snr[band] = (
                    snr_db
                )

                p_detect[band] = (
                    probability
                )

            else:

                observations[band] = int(
                    self.rng.random()
                    < BASE_FALSE_ALARM_RATE
                )

        return (
            observations,
            truths,
            {
                "mode": "STARE",
                "scanned": True,
                "noise_dbm": noise,
                "snr_db": snr,
                "p_detect": p_detect
            }
        )

    # SCAN STEP
    def step(
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
                "Simulation has finished."
            )

        # RETUNING
        if self.retune_remaining > 0:

            self.retune_remaining -= 1

            info = {
                "mode": "SCAN",
                "scanned": False,
                "retuning": True,

                "band":
                    self.target_band,

                "signal_power_dbm":
                    None,

                "noise_power_dbm":
                    None,

                "snr_db":
                    None,

                "p_detect":
                    None,

                "pw_us":
                    0.0,

                "aoa":
                    np.nan
            }

            self.t += 1

            done = (
                self.t
                >= self.n_slots
            )

            return (
                0,
                0,
                done,
                info
            )

        # DWELL
        if self.dwell_remaining > 0:

            band = self.current_band

            (
                observation,
                truth,
                info
            ) = self.observe_band(
                band
            )

            self.dwell_remaining -= 1

            info["dwell"] = True

            self.t += 1

            done = (
                self.t
                >= self.n_slots
            )

            return (
                observation,
                truth,
                done,
                info
            )

        # CHANGE BAND
        if (
            self.current_band
            is None
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

            self.t += 1

            done = (
                self.t
                >= self.n_slots
            )

            info = {
                "mode": "SCAN",
                "scanned": False,
                "retuning": True,

                "band":
                    requested_band,

                "signal_power_dbm":
                    None,

                "noise_power_dbm":
                    None,

                "snr_db":
                    None,

                "p_detect":
                    None,

                "pw_us":
                    0.0,

                "aoa":
                    np.nan
            }

            return (
                0,
                0,
                done,
                info
            )

        # NORMAL SCAN
        (
            observation,
            truth,
            info
        ) = self.observe_band(
            self.current_band
        )

        self.dwell_remaining = max(
            DWELL_SLOTS - 1,
            0
        )

        self.t += 1

        done = (
            self.t
            >= self.n_slots
        )

        return (
            observation,
            truth,
            done,
            info
        )