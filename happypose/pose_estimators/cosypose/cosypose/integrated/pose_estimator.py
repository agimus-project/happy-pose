import time
from collections import defaultdict
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import happypose.pose_estimators.cosypose.cosypose.utils.tensor_collection as tc
from happypose.pose_estimators.cosypose.cosypose.lib3d.cosypose_ops import (
    TCO_init_from_boxes,
    TCO_init_from_boxes_zup_autodepth,
)
from happypose.pose_estimators.cosypose.cosypose.utils.logging import get_logger
from happypose.pose_estimators.cosypose.cosypose.utils.timer import Timer
from happypose.pose_estimators.megapose.training.utils import CudaTimer, SimpleTimer
from happypose.toolbox.inference.pose_estimator import PoseEstimationModule
from happypose.toolbox.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from happypose.toolbox.inference.utils import filter_detections
from happypose.toolbox.utils.tensor_collection import PandasTensorCollection

logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PoseEstimator(PoseEstimationModule):
    """Performs inference for pose estimation."""

    def __init__(
        self,
        refiner_model: Optional[torch.nn.Module] = None,
        coarse_model: Optional[torch.nn.Module] = None,
        detector_model: Optional[torch.nn.Module] = None,
        # depth_refiner: Optional[DepthRefiner] = None,
        bsz_objects: int = 8,
        bsz_images: int = 256,
        # SO3_grid_size: int = 576,
    ) -> None:
        super().__init__()
        self.coarse_model = coarse_model
        self.refiner_model = refiner_model
        self.detector_model = detector_model
        # self.depth_refiner = depth_refiner
        self.bsz_objects = bsz_objects
        self.bsz_images = bsz_images

        # Load the SO3 grid if was passed in
        # if SO3_grid_size is not None:
        #    self.load_SO3_grid(SO3_grid_size)

        # load cfg and mesh_db from refiner model
        if self.refiner_model is not None:
            self.cfg = self.refiner_model.cfg
            self.mesh_db = self.refiner_model.mesh_db
        elif self.coarse_model is not None:
            self.cfg = self.coarse_model.cfg
            self.mesh_db = self.coarse_model.mesh_db
        else:
            msg = "At least one of refiner_model or  coarse_model must be specified."
            raise ValueError(msg)

        self.eval()

        self.keep_all_outputs = False
        self.keep_all_coarse_outputs = False
        self.refiner_outputs = None
        self.coarse_outputs = None
        self.debug_dict: dict = {}

    @torch.no_grad()
    def batched_model_predictions(self, model, images, K, obj_data, n_iterations=1):
        timer = Timer()
        timer.start()

        ids = torch.arange(len(obj_data))

        ds = TensorDataset(ids)
        dl = DataLoader(ds, batch_size=self.bsz_objects)

        preds = defaultdict(list)
        for (batch_ids,) in dl:
            timer.resume()
            obj_inputs = obj_data[batch_ids.numpy()]
            labels = obj_inputs.infos["label"].values
            im_ids = obj_inputs.infos["batch_im_id"].values
            images_ = images[im_ids]
            K_ = K[im_ids]
            TCO_input = obj_inputs.poses
            outputs = model(
                images=images_,
                K=K_,
                TCO=TCO_input,
                n_iterations=n_iterations,
                labels=labels,
            )
            timer.pause()
            for n in range(1, n_iterations + 1):
                iter_outputs = outputs[f"iteration={n}"]

                infos = obj_inputs.infos
                batch_preds = tc.PandasTensorCollection(
                    infos,
                    poses=iter_outputs["TCO_output"],
                    poses_input=iter_outputs["TCO_input"],
                    K_crop=iter_outputs["K_crop"],
                    boxes_rend=iter_outputs["boxes_rend"],
                    boxes_crop=iter_outputs["boxes_crop"],
                )
                preds[f"iteration={n}"].append(batch_preds)

        logger.debug(
            f"Pose prediction on {len(obj_data)} detections "
            f"(n_iterations={n_iterations}): {timer.stop()}",
        )
        preds = dict(preds)
        for k, v in preds.items():
            preds[k] = tc.concatenate(v)
        return preds

    def make_TCO_init(self, detections, K):
        K = K[detections.infos["batch_im_id"].values]
        boxes = detections.bboxes
        if self.coarse_model.cfg.init_method == "z-up+auto-depth":
            meshes = self.coarse_model.mesh_db.select(detections.infos["label"])
            points_3d = meshes.sample_points(2000, deterministic=True)
            TCO_init = TCO_init_from_boxes_zup_autodepth(boxes, points_3d, K)
        else:
            TCO_init = TCO_init_from_boxes(z_range=(1.0, 1.0), boxes=boxes, K=K)
        return tc.PandasTensorCollection(infos=detections.infos, poses=TCO_init)

    @torch.no_grad()
    def run_inference_pipeline(
        self,
        observation: ObservationTensor,
        detections: Optional[DetectionsType] = None,
        data_TCO_init: Optional[PandasTensorCollection] = None,
        run_detector: Optional[bool] = None,
        n_refiner_iterations: int = 1,
        n_coarse_iterations: int = 1,
        bsz_images: Optional[int] = None,
        bsz_objects: Optional[int] = None,
        coarse_estimates: Optional[PoseEstimatesType] = None,
        detection_th: float = 0.7,
        mask_th: float = 0.8,
        labels_to_keep: Optional[List[str]] = None,
    ) -> Tuple[PoseEstimatesType, dict]:
        timing_str = ""
        timer = SimpleTimer()
        timer.start()

        if bsz_images is not None:
            self.bsz_images = bsz_images

        if bsz_objects is not None:
            self.bsz_objects = bsz_objects

        if coarse_estimates is None:
            assert detections is not None or run_detector, (
                "You must either pass in `detections` or set run_detector=True"
            )
            if detections is None and run_detector:
                start_time = time.time()
                detections = self.forward_detection_model(
                    observation,
                    detection_th,
                    mask_th,
                )
                if torch.cuda.is_available():
                    detections = detections.cuda()
                else:
                    detections = detections
                elapsed = time.time() - start_time
                timing_str += f"detection={elapsed:.2f}, "

        preds = {}
        if data_TCO_init is None:
            assert detections is not None
            assert self.coarse_model is not None
            assert n_coarse_iterations > 0

            if labels_to_keep is not None:
                detections = filter_detections(detections, labels_to_keep)

            K = observation.K
            data_TCO_init = self.make_TCO_init(detections, K)
            coarse_preds, coarse_extra_data = self.forward_coarse_model(
                observation,
                data_TCO_init,
                n_iterations=n_coarse_iterations,
            )
            for n in range(1, n_coarse_iterations + 1):
                preds[f"coarse/iteration={n}"] = coarse_preds[f"iteration={n}"]
            data_TCO_coarse = coarse_preds[f"iteration={n_coarse_iterations}"]
        else:
            assert n_coarse_iterations == 0
            data_TCO = data_TCO_init
            preds["external_coarse"] = data_TCO
            data_TCO_coarse = data_TCO

        if n_refiner_iterations >= 1:
            assert self.refiner_model is not None
            refiner_preds, refiner_extra_data = self.forward_refiner(
                observation,
                data_TCO_coarse,
                n_iterations=n_refiner_iterations,
            )
            for n in range(1, n_refiner_iterations + 1):
                preds[f"refiner/iteration={n}"] = refiner_preds[f"iteration={n}"]
            data_TCO = refiner_preds[f"iteration={n_refiner_iterations}"]

        timer.stop()
        timing_str = f"total={timer.elapsed():.2f}, {timing_str}"

        extra_data: dict = {}
        extra_data["coarse"] = {"preds": data_TCO_coarse, "data": coarse_extra_data}
        extra_data["refiner_all_hypotheses"] = {
            "preds": preds,
            "data": refiner_extra_data,
        }
        extra_data["refiner"] = {"preds": data_TCO, "data": refiner_extra_data}
        extra_data["timing_str"] = timing_str
        extra_data["time"] = timer.elapsed()

        return data_TCO, extra_data

    def forward_detection_model(
        self,
        observation: ObservationTensor,
        detection_th: float = 0.7,
        mask_th: float = 0.8,
        *args: Any,
        **kwargs: Any,
    ) -> DetectionsType:
        """Runs the detector."""
        detections = self.detector_model.get_detections(
            observation=observation,
            one_instance_per_class=False,
            detection_th=detection_th,
            output_masks=False,
            mask_th=mask_th,
        )
        return detections

    @torch.no_grad()
    def forward_coarse_model(
        self,
        observation: ObservationTensor,
        data_TCO_input: PoseEstimatesType,
        n_iterations: int = 5,
        keep_all_outputs: bool = False,
        cuda_timer: bool = False,
    ) -> Tuple[dict, dict]:
        """Runs the refiner model for the specified number of iterations.

        Will actually use the batched_model_predictions to stay within
        batch size limit.

        Returns
        -------
            (preds, extra_data)

            preds:
                A dict with keys 'refiner/iteration={n}' for
                n=1,...,n_iterations

                Each value is a data_TCO_type.

            extra_data:
                A dict containing additional information such as timing

        """
        timer = Timer()
        timer.start()

        start_time = time.time()

        B = data_TCO_input.poses.shape[0]
        ids = torch.arange(B)
        ds = TensorDataset(ids)
        dl = DataLoader(ds, batch_size=self.bsz_objects)
        device = observation.images.device

        preds = defaultdict(list)
        all_outputs = []

        model_time = 0.0

        for batch_idx, (batch_ids,) in enumerate(dl):
            data_TCO_input_ = data_TCO_input[batch_ids]
            df_ = data_TCO_input_.infos
            TCO_input_ = data_TCO_input_.poses

            # Add some additional fields to df_
            df_["coarse_batch_idx"] = batch_idx
            df_["coarse_instance_idx"] = np.arange(len(df_))

            labels_ = df_["label"].tolist()
            batch_im_ids_ = torch.as_tensor(df_["batch_im_id"].values, device=device)

            images_ = observation.images[batch_im_ids_]
            K_ = observation.K[batch_im_ids_]
            if torch.cuda.is_available():
                timer_ = CudaTimer(enabled=cuda_timer)
            else:
                timer_ = SimpleTimer()
            timer_.start()
            outputs_ = self.coarse_model(
                images=images_,
                K=K_,
                TCO=TCO_input_,
                n_iterations=n_iterations,
                labels=labels_,
            )
            timer_.stop()
            model_time += timer_.elapsed()

            if keep_all_outputs:
                all_outputs.append(outputs_)

            # Collect the data
            for n in range(1, n_iterations + 1):
                iter_outputs = outputs_[f"iteration={n}"]

                infos = df_
                batch_preds = PandasTensorCollection(
                    infos,
                    poses=iter_outputs.TCO_output,
                    poses_input=iter_outputs.TCO_input,
                    K_crop=iter_outputs.K_crop,
                    K=iter_outputs.K,
                    boxes_rend=iter_outputs.boxes_rend,
                    boxes_crop=iter_outputs.boxes_crop,
                )

                preds[f"iteration={n}"].append(batch_preds)

        for k, v in preds.items():
            preds[k] = tc.concatenate(v)

        timer.stop()

        elapsed = time.time() - start_time

        extra_data = {
            "n_iterations": n_iterations,
            "outputs": all_outputs,
            "model_time": model_time,
            "time": elapsed,
        }

        return preds, extra_data

    @torch.no_grad()
    def forward_refiner(
        self,
        observation: ObservationTensor,
        data_TCO_input: PoseEstimatesType,
        n_iterations: int = 5,
        keep_all_outputs: bool = False,
        cuda_timer: bool = False,
    ) -> Tuple[dict, dict]:
        """Runs the refiner model for the specified number of iterations.

        Will actually use the batched_model_predictions to stay within
        batch size limit.

        Returns
        -------
            (preds, extra_data)

            preds:
                A dict with keys 'refiner/iteration={n}' for
                n=1,...,n_iterations

                Each value is a data_TCO_type.

            extra_data:
                A dict containing additional information such as timing

        """
        timer = Timer()
        timer.start()

        start_time = time.time()

        B = data_TCO_input.poses.shape[0]
        ids = torch.arange(B)
        ds = TensorDataset(ids)
        dl = DataLoader(ds, batch_size=self.bsz_objects)
        device = observation.images.device

        preds = defaultdict(list)
        all_outputs = []

        model_time = 0.0

        for batch_idx, (batch_ids,) in enumerate(dl):
            data_TCO_input_ = data_TCO_input[batch_ids]
            df_ = data_TCO_input_.infos
            TCO_input_ = data_TCO_input_.poses

            # Add some additional fields to df_
            df_["refiner_batch_idx"] = batch_idx
            df_["refiner_instance_idx"] = np.arange(len(df_))

            labels_ = df_["label"].tolist()
            batch_im_ids_ = torch.as_tensor(df_["batch_im_id"].values, device=device)

            images_ = observation.images[batch_im_ids_]
            K_ = observation.K[batch_im_ids_]
            if torch.cuda.is_available():
                timer_ = CudaTimer(enabled=cuda_timer)
            else:
                timer_ = SimpleTimer()
            timer_.start()
            outputs_ = self.refiner_model(
                images=images_,
                K=K_,
                TCO=TCO_input_,
                n_iterations=n_iterations,
                labels=labels_,
            )
            timer_.stop()
            model_time += timer_.elapsed()

            if keep_all_outputs:
                all_outputs.append(outputs_)

            # Collect the data
            for n in range(1, n_iterations + 1):
                iter_outputs = outputs_[f"iteration={n}"]

                infos = df_
                batch_preds = PandasTensorCollection(
                    infos,
                    poses=iter_outputs.TCO_output,
                    poses_input=iter_outputs.TCO_input,
                    K_crop=iter_outputs.K_crop,
                    K=iter_outputs.K,
                    boxes_rend=iter_outputs.boxes_rend,
                    boxes_crop=iter_outputs.boxes_crop,
                )

                preds[f"iteration={n}"].append(batch_preds)

        for k, v in preds.items():
            preds[k] = tc.concatenate(v)

        timer.stop()

        elapsed = time.time() - start_time

        extra_data = {
            "n_iterations": n_iterations,
            "outputs": all_outputs,
            "model_time": model_time,
            "time": elapsed,
        }

        logger.debug(
            f"Pose prediction on {B} poses (n_iterations={n_iterations}):"
            f" {timer.stop()}",
        )

        return preds, extra_data
