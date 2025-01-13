import os
from pathlib import Path

# Standard Library
from typing import Dict

import pytest
import torch

# Third Party
from omegaconf import OmegaConf

from happypose.pose_estimators.megapose.config import LOCAL_DATA_DIR, RESULTS_DIR
from happypose.pose_estimators.megapose.evaluation.bop import run_evaluation
from happypose.pose_estimators.megapose.evaluation.eval_config import (
    BOPEvalConfig,
    EvalConfig,
    FullEvalConfig,
    HardwareConfig,
)
from happypose.pose_estimators.megapose.evaluation.evaluation import get_save_dir
from happypose.pose_estimators.megapose.scripts.run_full_megapose_eval import (
    create_eval_cfg,
)
from happypose.toolbox.utils.distributed import get_rank
from happypose.toolbox.utils.logging import get_logger

logger = get_logger(__name__)


class TestCosyPoseEvaluation:
    """Unit tests for CosyPose evaluation."""

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        configuration = {
            "detector_run_id": "bop_pbr",
            "coarse_run_id": "coarse-bop-ycbv-pbr--724183",
            "refiner_run_id": "refiner-bop-ycbv-pbr--604090",
            "ds_names": ["ycbv.bop19"],
            "result_id": "ycbv-debug",
            "detection_coarse_types": [["detector", "S03_grid"]],
            "skip_inference": True,
            "run_bop_eval": True,
        }

        conf = OmegaConf.create(configuration)

        cfg: FullEvalConfig = OmegaConf.structured(FullEvalConfig)

        cfg.hardware = HardwareConfig(
            n_cpus=int(os.environ.get("N_CPUS", 10)),
            n_gpus=int(os.environ.get("WORLD_SIZE", 1)),
        )

        cfg = OmegaConf.merge(cfg, conf)

        cfg.save_dir = RESULTS_DIR / cfg.result_id

        self.cfg = cfg

        # Iterate over each dataset
        for ds_name in self.cfg.ds_names:
            # create the EvalConfig objects that we will call `run_eval` on
            eval_configs: Dict[str, EvalConfig] = {}
            for (
                detection_type,
                coarse_estimation_type,
            ) in self.cfg.detection_coarse_types:
                name, cfg_ = create_eval_cfg(
                    self.cfg,
                    detection_type,
                    coarse_estimation_type,
                    ds_name,
                )
                eval_configs[name] = cfg_

        self.eval_cfg = eval_configs

    def test_config(self):
        assert self.eval_cfg is not None
        assert len(self.eval_cfg) == 1
        assert self.cfg["detector_run_id"] == "bop_pbr", (
            "Error: detector_run_id is not correct"
        )
        assert self.cfg["coarse_run_id"] == "coarse-bop-ycbv-pbr--724183", (
            "Error: coarse_run_id is not correct"
        )
        assert self.cfg["refiner_run_id"] == "refiner-bop-ycbv-pbr--604090", (
            "Error: refiner_run_id is not correct"
        )
        assert self.cfg["ds_name"] == "ycbv.bop19", "Error: ds_name is not correct"
        assert self.cfg["inference"]["detection_type"] == "detector", (
            "Error: detection_type is not correct"
        )
        assert self.cfg["inference"]["coarse_estimation_type"] == "SO3_grid", (
            "Error: coarse_estimation_type is not correct"
        )
        assert self.cfg["inference"]["SO3_grid_size"] == 576, (
            "Error: SO3_grid_size is not correct"
        )
        assert self.cfg["inference"]["n_refiner_iterations"] == 5, (
            "Error: n_refiner_iterations is not correct"
        )
        assert self.cfg["inference"]["n_pose_hypotheses"] == 5, (
            "Error: n_pose_hypotheses is not correct"
        )
        assert not self.cfg["inference"]["run_depth_refiner"], (
            "Error: run_depth_refiner is not correct"
        )
        assert self.cfg["inference"]["depth_refiner"] is None, (
            "Error: depth_refiner is not correct"
        )
        assert self.cfg["inference"]["bsz_objects"] == 16, (
            "Error: bsz_objects is not correct"
        )
        assert self.cfg["inference"]["bsz_images"] == 288, (
            "Error: bsz_images is not correct"
        )
        assert self.cfg["result_id"] == "ycbv-debug", "Error: result_id is not correct"
        assert self.cfg["n_dataloader_workers"] == 8, (
            "Error: n_dataloader_workers is not correct"
        )
        assert self.cfg["n_rendering_workers"] == 8, (
            "Error: n_rendering_workers is not correct"
        )
        assert self.cfg["n_frames"] is None, "Error: n_frames is not correct"
        assert self.cfg["batch_size"] == 1, "Error: batch_size is not correct"
        assert self.cfg["save_dir"] == f"{LOCAL_DATA_DIR}/results/ycbv-debug", (
            "Error: save_dir is not correct"
        )
        assert self.cfg["bsz_images"] == 256, "Error: bsz_images is not correct"
        assert self.cfg["bsz_objects"] == 16, "Error: bsz_objects is not correct"
        assert self.cfg["skip_inference"], "Error: skip_inference is not correct"
        assert self.cfg["skip_evaluation"], "Error: skip_evaluation is not correct"
        assert self.cfg["global_batch_size"] is None, (
            "Error: global_batch_size is not correct"
        )
        assert self.cfg["hardware"]["n_cpus"] == 10, "Error: n_cpus is not correct"
        assert self.cfg["hardware"]["n_gpus"] == 1, "Error: n_gpus is not correct"
        assert not self.cfg["debug"], "Error: debug is not correct"
        assert self.cfg["detection_coarse_types"] == [["detector", "S03_grid"]], (
            "Error: detection_coarse_types is not correct"
        )
        assert self.cfg["ds_names"] == ["ycbv.bop19"], "Error: ds_names is not correct"
        assert self.cfg["run_bop_eval"], "Error: run_bop_eval is not correct"
        assert not self.cfg["eval_coarse_also"], (
            "Error: eval_coarse_also is not correct"
        )
        assert not self.cfg["convert_only"], "Error: convert_only is not correct"

    # TODO
    # Rajouter un test pour save_dir ?
    # Modifier ensuite pour que le path soit un path temporaire ?
    def test_evaluation_existing_results(self):
        if torch.cuda.is_available():
            bop_eval_cfgs = []
            for ds_name in self.cfg.ds_names:
                # For each eval_cfg run the evaluation.
                # Note that the results get saved to disk
                for _save_key, eval_cfg in self.eval_cfg.items():
                    results_dir = get_save_dir(eval_cfg)
                    pred_keys = ["refiner/final"]
                    if eval_cfg.inference.run_depth_refiner:
                        pred_keys.append("depth_refiner")
                    eval_out = {
                        "results_path": results_dir / "results.pth.tar",
                        "pred_keys": pred_keys,
                        "save_dir": results_dir,
                    }
                    # Test results_dir and eval_out here
                    assert Path(
                        eval_out["results_path"],
                    ).is_file(), f"The file {eval_out['results_path']} doesn't exist"

                    # Run the bop eval for each type of prediction
                    if self.cfg.run_bop_eval and get_rank() == 0:
                        bop_eval_keys = {"refiner/final", "depth_refiner"}
                        bop_eval_keys = bop_eval_keys.intersection(
                            set(eval_out["pred_keys"])
                        )

                        for method in bop_eval_keys:
                            if "bop19" not in ds_name:
                                continue

                            bop_eval_cfg = BOPEvalConfig(
                                results_path=eval_out["results_path"],
                                dataset=ds_name,
                                split="test",
                                eval_dir=eval_out["save_dir"] / "bop_evaluation",
                                method=method,
                                convert_only=False,
                                use_post_score=False,
                            )
                            bop_eval_cfgs.append(bop_eval_cfg)

            assert bop_eval_cfg.results_path == eval_out["results_path"]
            assert bop_eval_cfg.dataset == "ycbv.bop19"

            if get_rank() == 0:
                if self.cfg.run_bop_eval:
                    for bop_eval_cfg in bop_eval_cfgs:
                        scores_pose_path, _ = run_evaluation(bop_eval_cfg)
                        assert Path(
                            LOCAL_DATA_DIR
                            / "bop_eval_outputs"
                            / f"refiner-final_{bop_eval_cfg.dataset.split('.')[0]}-{bop_eval_cfg.split}"
                            / "scores_bop19.json"
                        ).is_file()
                        # assert scores_pose_path.is_file()

            logger.info(f"Process {get_rank()} reached end of script")
        else:
            pytest.skip("Evaluation is not tested without GPU")

    # TODO : Run the inference, then use the results for evaluation
    """
    # The inference shouldn't be tested in this function?
    # This runs the inference
    @pytest.fixture(autouse=True)
    def eval_out(self, setUp):
        for _save_key, eval_cfg in self.eval_cfg.items():
        # Run the inference
            self.eval_out = run_eval(eval_cfg)

    def test_evaluation_inference(self, eval_out):
        bop_eval_cfgs = []
        for ds_name in self.cfg.ds_names:
            bop_eval_keys = {"refiner/final", "depth_refiner"}
            bop_eval_keys = bop_eval_keys.intersection(set(eval_out["pred_keys"]))

            for method in bop_eval_keys:
                if "bop19" not in ds_name:
                    continue

                bop_eval_cfg = BOPEvalConfig(
                    results_path=eval_out["results_path"],
                    dataset=ds_name,
                    split="test",
                    eval_dir=eval_out["save_dir"] / "bop_evaluation",
                    method=method,
                    convert_only=False,
                    use_post_score=False,
                )
                bop_eval_cfgs.append(bop_eval_cfg)

        if get_rank() == 0:
            if self.cfg.run_bop_eval:
                for bop_eval_cfg in bop_eval_cfgs:
                    run_evaluation(bop_eval_cfg)

        logger.info(f"Process {get_rank()} reached end of script")
        """
