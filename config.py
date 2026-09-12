REPO_ID = "alan-turing-institute/turing-synthetic-radar-dataset"

INPUT_RECEIVER_MODE = "stare"

N_FILES_TO_LOAD = 15
MAX_ROWS_PER_FILE = 500_000

# TSRD PDW frequency is in MHz.
FREQ_MIN_MHZ = 0.0
FREQ_MAX_MHZ = 18_000.0

# TSRD scan receiver: 500 MHz bandwidth, 500 MHz frequency steps.
SCAN_BANDWIDTH_MHZ = 500.0
SCAN_STEP_MHZ = 500.0
N_BANDS = int((FREQ_MAX_MHZ - FREQ_MIN_MHZ) / SCAN_STEP_MHZ)

# Simulation
SIMULATION_DURATION_S = 10.0
SIMULATION_DURATION_US = SIMULATION_DURATION_S * 1_000_000.0

TARGET_SLOTS = 200_000
SLOT_DURATION_US = SIMULATION_DURATION_US / TARGET_SLOTS
MAX_SLOTS = 2_000_000

# Receiver/noise model
NOISE_FLOOR_DBM = -100.0
NOISE_STD_DB = 2.0
RECEIVER_SENSITIVITY_DBM = -110.0

DETECTION_SNR_50_DB = 6.0
DETECTION_SLOPE = 0.8

# Dynamic false-alarm model
FALSE_ALARM_THRESHOLD_DBM = -100.0
FALSE_ALARM_SLOPE = 1.0
BASE_FALSE_ALARM_RATE = 0.01
MAX_FALSE_ALARM_RATE = 0.10

# Receiver timing
DWELL_SLOTS = 3
RETUNE_SLOTS = 2

# Pulse-width contribution to detection probability
MIN_PW_US = 0.1
MAX_PW_US = 1000.0
REFERENCE_PW_US = 1.0

# Optional interference model
INTERFERENCE_PROB = 0.05
INTERFERENCE_PENALTY_DB = 6.0

# Extra random signal loss after the receiver has successfully
# tuned to the correct band.
SIGNAL_DROP_PROB = 0.0
