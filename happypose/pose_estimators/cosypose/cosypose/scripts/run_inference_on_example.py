# Standard Library
import argparse
import os
from pathlib import Path

# Third Party
import torch

from happypose.pose_estimators.cosypose.cosypose.integrated.pose_estimator import (
    PoseEstimator,
)

# CosyPose
from happypose.pose_estimators.cosypose.cosypose.utils.cosypose_wrapper import (
    CosyPoseWrapper,
)

# HappyPose
from happypose.toolbox.datasets.object_dataset import RigidObjectDataset
from happypose.toolbox.inference.example_inference_utils import (
    load_detections,
    load_object_data,
    load_observation_example,
    make_detections_visualization,
    make_example_object_dataset,
    make_poses_visualization,
    save_predictions,
)
from happypose.toolbox.inference.types import DetectionsType, ObservationTensor
from happypose.toolbox.inference.utils import filter_detections, load_detector
from happypose.toolbox.utils.logging import get_logger, set_logging_level

logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_pose_estimator(dataset_to_use: str, object_dataset: RigidObjectDataset):
    # TODO: remove this wrapper from code base
    cosypose = CosyPoseWrapper(
        dataset_name=dataset_to_use, object_dataset=object_dataset, n_workers=1
    )

    return cosypose.pose_predictor


def run_inference(
    pose_estimator: PoseEstimator,
    observation: ObservationTensor,
    detections: DetectionsType,
) -> None:
    observation.to(device)

    data_TCO, extra_data = pose_estimator.run_inference_pipeline(
        observation=observation, detections=detections, n_refiner_iterations=3
    )
    print("Timings:")
    print(extra_data["timing_str"])

    return data_TCO.cpu()


def main():
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("example_name")
    parser.add_argument("--dataset", type=str, default="hope")
    parser.add_argument("--run-detections", action="store_true")
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--vis-detections", action="store_true")
    parser.add_argument("--vis-poses", action="store_true")
    args = parser.parse_args()

    data_dir = os.getenv("HAPPYPOSE_DATA_DIR")
    assert data_dir, "Set HAPPYPOSE_DATA_DIR env variable"
    example_dir = Path(data_dir) / "examples" / args.example_name
    assert example_dir.exists(), (
        "Example {args.example_name} not available, follow download instructions"
    )
    # dataset_to_use = args.dataset  # hope/tless/ycbv

    # Load data
    detections = load_detections(example_dir).to(device)
    object_dataset = make_example_object_dataset(example_dir)
    rgb, depth, camera_data = load_observation_example(example_dir, load_depth=False)
    # TODO: cosypose forward does not work if depth is loaded detection
    # contrary to megapose
    observation = ObservationTensor.from_numpy(rgb, depth=None, K=camera_data.K).to(
        device
    )

    # Load models
    pose_estimator = setup_pose_estimator(args.dataset, object_dataset)

    if args.run_detections:
        # TODO: hardcoded detector
        detector = load_detector(run_id="detector-bop-hope-pbr--15246", device=device)
        # Masks are not used for pose prediction, but are computed by Mask-RCNN anyway
        detections = detector.get_detections(observation, output_masks=True)
        available_labels = [obj.label for obj in object_dataset.list_objects]
        detections = filter_detections(detections, available_labels)
    else:
        detections = load_detections(example_dir).to(device)

    if args.run_inference:
        output = run_inference(pose_estimator, observation, detections)
        save_predictions(output, example_dir)

    if args.vis_detections:
        make_detections_visualization(rgb, detections, example_dir)

    if args.vis_poses:
        if args.run_inference:
            out_filename = "object_data_inf.json"
        else:
            out_filename = "object_data.json"
        object_datas = load_object_data(example_dir / "outputs" / out_filename)
        make_poses_visualization(
            rgb, object_dataset, object_datas, camera_data, example_dir
        )


if __name__ == "__main__":
    main()
