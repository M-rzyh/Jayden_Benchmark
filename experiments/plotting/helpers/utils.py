ENVIRONMENTS_MAP = {
    "MOHopperDR-v5": ["MOHopperDefault-v5","MOHopperLight-v5","MOHopperHeavy-v5","MOHopperSlippery-v5","MOHopperLowDamping-v5","MOHopperHard-v5"],
    "MOHalfCheetahDR-v5": ["MOHalfCheetahDefault-v5","MOHalfCheetahLight-v5","MOHalfCheetahHeavy-v5","MOHalfCheetahSlippery-v5","MOHalfCheetahHard-v5"],
    "MOHumanoidDR-v5": ["MOHumanoidDefault-v5","MOHumanoidLight-v5","MOHumanoidHeavy-v5","MOHumanoidLowDamping-v5","MOHumanoidHard-v5"],
    "MOLunarLanderDR-v0": ["MOLunarLanderDefault-v0","MOLunarLanderHighGravity-v0","MOLunarLanderWindy-v0","MOLunarLanderTurbulent-v0","MOLunarLanderHard-v0"],
}

ALGORITHMS = [
    'MORL-D(MOSAC)-SB+PSA', 
    'MORL-D(MOSAC)-SB', 
    'GPI-PD Continuous Action', 
    'GPI-LS Continuous Action', 
    'PGMORL', 
    'CAPQL', 
    'PCN continuous action',
    'SAC Continuous Action',
]

ALGORITHMS_NAME_MAP = {
    'PCN continuous action': 'PCN',
    'PGMORL': 'PGMORL',
    'CAPQL': 'CAPQL',
    'GPI-LS Continuous Action': 'GPI-LS',
    'GPI-PD Continuous Action': 'GPI-PD',
    'MORL-D(MOSAC)-SB': 'MORL-D(SB)',
    'MORL-D(MOSAC)-SB+PSA': 'MORL-D(SB+PSA)',
    'SAC Continuous Action': 'SAC',
}