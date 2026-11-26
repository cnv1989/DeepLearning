"""Model/hardware configurations.

N      context length (how many previous characters the model sees)
D      embedding size per character (power of two)
H      hidden neurons
S1     right-shift used to requantise the hidden layer (h = round(relu(a) / 2**S1))
E_BITS signed bits per embedding value
H_BITS unsigned bits per hidden activation
B_BITS signed bits per bias
ACC_W  accumulator width in hardware
NOISE_BITS  width of the LFSR noise added to logits when sampling is enabled
"""
V = 32

CONFIGS = {
    # Small enough to be a realistic Tiny Tapeout multi-tile project.
    "tt": dict(N=3, D=4, H=16, S1=2, E_BITS=4, H_BITS=4, B_BITS=10, ACC_W=16, NOISE_BITS=3),
    # Bigger model for an FPGA (iCE40 UP5K / Artix-7 class).
    "fpga": dict(N=6, D=8, H=48, S1=3, E_BITS=4, H_BITS=4, B_BITS=10, ACC_W=16, NOISE_BITS=3),
}


def get(name: str) -> dict:
    cfg = dict(CONFIGS[name])
    cfg["V"] = V
    cfg["ND"] = cfg["N"] * cfg["D"]
    cfg["name"] = name
    assert cfg["D"] & (cfg["D"] - 1) == 0, "D must be a power of two"
    assert cfg["S1"] >= 1
    return cfg
