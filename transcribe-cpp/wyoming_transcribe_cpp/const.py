"""Shared constants."""

PORT = 10380
MODELS_DIR = "/data/models"

# Exit code for a bootstrap/configuration failure (EX_CONFIG from sysexits.h).
# Deterministic — it would repeat on every restart — so the s6 finish handler
# halts the container with 0, keeping the Supervisor watchdog from
# restart-looping a broken config. Any other non-zero exit is a runtime crash
# and *is* worth restarting.
EXIT_BOOTSTRAP = 78

# GGUF precisions ordered smallest -> largest. Quant resolution walks this
# list upward from the requested precision (never downward — a smaller
# precision than asked for would silently degrade quality).
QUANT_ORDER = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16", "F32"]

# config.yaml option values (lowercase) -> GGUF precision names.
QUANT_ALIASES = {q.lower(): q for q in QUANT_ORDER}
