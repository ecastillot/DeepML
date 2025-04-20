import os
import sys
import joblib
import numpy as np
import seisbench.data as sbd
import seisbench.generate as sbg

root = "/groups/igonin/.seisbench"
os.environ["SEISBENCH_CACHE_ROOT"] = root


def load_scaler(scaler_path):
    return joblib.load(scaler_path)

def build_normalization(key="X"):
    return sbg.Normalize(
        detrend_axis=0,
        amp_norm_type="peak",
        eps=1e-8,
        key=key,
    )

def build_detection_label(key=("X", "y_detection")):
    return sbg.DetectionLabeller(
        p_phases="trace_p_arrival_sample",
        s_phases="trace_s_arrival_sample",
        factor=1.5,
        key=key,
    )

def build_ps_probabilistic_label(key=("X", "y_picks")):
    return sbg.ProbabilisticLabeller(
        label_columns=["trace_p_arrival_sample", "trace_s_arrival_sample"],
        sigma=50,
        dim=-2,
        key=key,
    )

def build_magnitude_labeler(generator, scaler):
    @generator.augmentation
    def magnitude_labeler(state_dict):
        _, metadata = state_dict["X"]
        norm_mag = scaler.transform(np.array([metadata["source_magnitude"]]).reshape(-1, 1))
        state_dict["y_magnitude"] = [norm_mag]

def create_generator(
    dataset_name="TXED",
    scaler_path="/home/edc240000/DeepML/tests/magnitude_scaler.pkl",
    add_detection=True,
    add_probabilistic=False,
    add_magnitude=True,
):
    
    # Load dataset
    if dataset_name == "TXED":
        data = sbd.TXED()
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported yet.")

    # Create generator
    generator = sbg.GenericGenerator(data)

    # Add augmentations
    augmentations = [build_normalization()]

    if add_detection:
        augmentations.append(build_detection_label())
    if add_probabilistic:
        augmentations.append(build_ps_probabilistic_label())

    generator.add_augmentations(augmentations)

    # Add magnitude labeler
    if add_magnitude:
        scaler = load_scaler(scaler_path)
        build_magnitude_labeler(generator, scaler)

    return generator
