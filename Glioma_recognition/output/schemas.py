CONTRACT_VERSION = "0.1"

PREDICTION_REQUIRED_KEYS = {
    "AccessionNumber",
    "IsNotHumanBodyProb",
    "IsStitchedProb",
    "ProcessingTime_ms",
    "SegmentationMaskURI",
    "Prediction",
    "Interpretation",
}

MASK_KEYS = {"core", "flair"}

DUPLICATE_KEYS = {"StudyUID", "StudyUID_dup", "PairProb"}

