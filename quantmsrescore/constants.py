PRIMARY_SCORE_IMPUTED = "quantms:primary_score_imputed"

MS2PIP_FEATURES = {
    "MS2PIP:SpecPearson": "spec_pearson",
    "MS2PIP:SpecCosineNorm": "cos_norm",
    "MS2PIP:SpecPearsonNorm": "spec_pearson_norm",
    "MS2PIP:DotProd": "dotprod",
    "MS2PIP:IonBPearsonNorm": "ionb_pearson_norm",
    "MS2PIP:IonYPearsonNorm": "iony_pearson_norm",
    "MS2PIP:SpecMseNorm": "spec_mse_norm",
    "MS2PIP:IonBMseNorm": "ionb_mse_norm",
    "MS2PIP:IonYMseNorm": "iony_mse_norm",
    "MS2PIP:MinAbsDiffNorm": "min_abs_diff_norm",
    "MS2PIP:MaxAbsDiffNorm": "max_abs_diff_norm",
    "MS2PIP:AbsDiffQ1Norm": "abs_diff_Q1_norm",
    "MS2PIP:AbsDiffQ2Norm": "abs_diff_Q2_norm",
    "MS2PIP:AbsDiffQ3Norm": "abs_diff_Q3_norm",
    "MS2PIP:MeanAbsDiffNorm": "mean_abs_diff_norm",
    "MS2PIP:StdAbsDiffNorm": "std_abs_diff_norm",
    "MS2PIP:IonBMinAbsDiffNorm": "ionb_min_abs_diff_norm",
    "MS2PIP:IonBMaxAbsDiffNorm": "ionb_max_abs_diff_norm",
    "MS2PIP:IonBAbsDiffQ1Norm": "ionb_abs_diff_Q1_norm",
    "MS2PIP:IonBAbsDiffQ2Norm": "ionb_abs_diff_Q2_norm",
    "MS2PIP:IonBAbsDiffQ3Norm": "ionb_abs_diff_Q3_norm",
    "MS2PIP:IonBMeanAbsDiffNorm": "ionb_mean_abs_diff_norm",
    "MS2PIP:IonBStdAbsDiffNorm": "ionb_std_abs_diff_norm",
    "MS2PIP:IonYMinAbsDiffNorm": "iony_min_abs_diff_norm",
    "MS2PIP:IonYMaxAbsDiffNorm": "iony_max_abs_diff_norm",
    "MS2PIP:IonYAbsDiffQ1Norm": "iony_abs_diff_Q1_norm",
    "MS2PIP:IonYAbsDiffQ2Norm": "iony_abs_diff_Q2_norm",
    "MS2PIP:IonYAbsDiffQ3Norm": "iony_abs_diff_Q3_norm",
    "MS2PIP:IonYMeanAbsDiffNorm": "iony_mean_abs_diff_norm",
    "MS2PIP:IonYStdAbsDiffNorm": "iony_std_abs_diff_norm",
    "MS2PIP:DotProdNorm": "dotprod_norm",
    "MS2PIP:DotProdIonBNorm": "dotprod_ionb_norm",
    "MS2PIP:DotProdIonYNorm": "dotprod_iony_norm",
    "MS2PIP:CosIonBNorm": "cos_ionb_norm",
    "MS2PIP:CosIonYNorm": "cos_iony_norm",
    "MS2PIP:IonBPearson": "ionb_pearson",
    "MS2PIP:IonYPearson": "iony_pearson",
    "MS2PIP:SpecSpearman": "spec_spearman",
    "MS2PIP:IonBSpearman": "ionb_spearman",
    "MS2PIP:IonYSpearman": "iony_spearman",
    "MS2PIP:SpecMse": "spec_mse",
    "MS2PIP:IonBMse": "ionb_mse",
    "MS2PIP:IonYMse": "iony_mse",
    "MS2PIP:MinAbsDiffIonType": "min_abs_diff_iontype",
    "MS2PIP:MaxAbsDiffIonType": "max_abs_diff_iontype",
    "MS2PIP:MinAbsDiff": "min_abs_diff",
    "MS2PIP:MaxAbsDiff": "max_abs_diff",
    "MS2PIP:AbsDiffQ1": "abs_diff_Q1",
    "MS2PIP:AbsDiffQ2": "abs_diff_Q2",
    "MS2PIP:AbsDiffQ3": "abs_diff_Q3",
    "MS2PIP:MeanAbsDiff": "mean_abs_diff",
    "MS2PIP:StdAbsDiff": "std_abs_diff",
    "MS2PIP:IonBMinAbsDiff": "ionb_min_abs_diff",
    "MS2PIP:IonBMaxAbsDiff": "ionb_max_abs_diff",
    "MS2PIP:IonBAbsDiffQ1": "ionb_abs_diff_Q1",
    "MS2PIP:IonBAbsDiffQ2": "ionb_abs_diff_Q2",
    "MS2PIP:IonBAbsDiffQ3": "ionb_abs_diff_Q3",
    "MS2PIP:IonBMeanAbsDiff": "ionb_mean_abs_diff",
    "MS2PIP:IonBStdAbsDiff": "ionb_std_abs_diff",
    "MS2PIP:IonYMinAbsDiff": "iony_min_abs_diff",
    "MS2PIP:IonYMaxAbsDiff": "iony_max_abs_diff",
    "MS2PIP:IonYAbsDiffQ1": "iony_abs_diff_Q1",
    "MS2PIP:IonYAbsDiffQ2": "iony_abs_diff_Q2",
    "MS2PIP:IonYAbsDiffQ3": "iony_abs_diff_Q3",
    "MS2PIP:IonYMeanAbsDiff": "iony_mean_abs_diff",
    "MS2PIP:IonYStdAbsDiff": "iony_std_abs_diff",
    "MS2PIP:DotProdIonB": "dotprod_ionb",
    "MS2PIP:DotProdIonY": "dotprod_iony",
    "MS2PIP:Cos": "cos",
    "MS2PIP:CosIonB": "cos_ionb",
    "MS2PIP:CosIonY": "cos_iony",
}

DEEPLC_FEATURES = {
    "DeepLC:ObservedRetentionTime": "observed_retention_time",
    "DeepLC:PredictedRetentionTime": "predicted_retention_time",
    "DeepLC:RtDiff": "rt_diff",
    "DeepLC:ObservedRetentionTimeBest": "observed_retention_time_best",
    "DeepLC:PredictedRetentionTimeBest": "predicted_retention_time_best",
    "DeepLC:RtDiffBest": "rt_diff_best",
}

QUANTMS_FEATURES = {
    "Quantms:Snr": "snr",
    "Quantms:SpectralEntropy": "spectral_entropy",
    "Quantms:FracTICinTop10Peaks": "fraction_tic_top_10",
    "Quantms:WeightedStdMz": "weighted_std_mz",
}

SUPPORTED_MODELS_MS2PIP = {
    "HCD": [
        "HCD2019",  # HCD from 2019
        "HCD2021",  # Default model
        "Immuno-HCD",  # Immuno-HCD
        "HCDch2",  # HCD with charge 2
        "TMT",  # TMT 10-plex
        "iTRAQ",  # iTRAQ
        "iTRAQphospho",  # iTRAQ phospho
    ],
    "CID": [
        "CID",  # Collision-induced dissociation
        "CIDch2",  # CID with charge 2
        "CID-TMT",  # CID-TMT
    ],
}

# This is the list of disassociation methods that are supported by OPENMS.
# This list is a path for release 3.3.0 of OpenMS.
OPENMS_DISSOCIATION_METHODS_PATCH_3_3_0 = [
    {
        "CID": "Collision-induced dissociation (MS:1000133) (also CAD; parent term, but unless otherwise stated often used as synonym for trap-type CID)"
    },
    {"PSD": "Post-source decay."},
    {"PD": "Plasma desorption."},
    {"SID": "Surface-induced dissociation."},
    {"BIRD": "Blackbody infrared radiative dissociation."},
    {"ECD": "Electron capture dissociation (MS:1000250)"},
    {"IMD": "Infrared multiphoton dissociation."},
    {"SORI": "Sustained off-resonance irradiation."},
    {"HCID": "High-energy collision-induced dissociation."},
    {"LCID": "Low-energy collision-induced dissociation."},
    {"PHD": "Photodissociation."},
    {"ETD": "Electron transfer dissociation."},
    {"ETciD": "Electron transfer and collision-induced dissociation (MS:1003182)"},
    {"EThcD": "Electron transfer and higher-energy collision dissociation (MS:1002631)"},
    {"PQD": "Pulsed q dissociation (MS:1000599)"},
    {"TRAP": "trap-type collision-induced dissociation (MS:1002472)"},
    {"HCD": "beam-type collision-induced dissociation (MS:1000422)"},
    {"INSOURCE": "in-source collision-induced dissociation (MS:1001880)"},
    {"LIFT": "Bruker proprietary method (MS:1002000)"},
]

OPENMS_DISSOCIATION_METHODS_PATCH_3_1_0 = [
    {"CID": "Collision-induced dissociation"},
    {"PSD": "Post-source decay"},
    {"PD": "Plasma desorption"},
    {"SID": "Surface-induced dissociation"},
    {"BIRD": "Blackbody infrared radiative dissociation"},
    {"ECD": "Electron capture dissociation"},
    {"IMD": "Infrared multiphoton dissociation"},
    {"SORI": "Sustained off-resonance irradiation"},
    {"HCID": "High-energy collision-induced dissociation"},
    {"LCID": "Low-energy collision-induced dissociation"},
    {"PHD": "Photodissociation"},
    {"ETD": "Electron transfer dissociation"},
    {"PQD": "Pulsed q dissociation"},
    {"TRAP": "trap-type collision-induced dissociation"},
    {"HCD": "beam-type collision-induced dissociation"},
    {"INSOURCE": "in-source collision-induced dissociation"},
    {"LIFT": "Bruker proprietary method"},
]
