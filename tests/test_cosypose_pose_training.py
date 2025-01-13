import functools
import os
import time
from collections import defaultdict

import numpy as np
import pytest
import torch
import torch.distributed as dist
import yaml
from omegaconf import OmegaConf
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter
from tqdm import tqdm

from happypose.pose_estimators.cosypose.cosypose.config import EXP_DIR
from happypose.pose_estimators.cosypose.cosypose.training.pose_forward_loss import (
    h_pose,
)
from happypose.pose_estimators.cosypose.cosypose.training.pose_models_cfg import (
    check_update_config,
    create_pose_model_cosypose,
)
from happypose.pose_estimators.cosypose.cosypose.training.train_pose import (
    make_eval_bundle,
    run_eval,
)
from happypose.pose_estimators.cosypose.cosypose.utils.distributed import (
    get_world_size,
    init_distributed_mode,
    reduce_dict,
    sync_model,
)
from happypose.toolbox.datasets.datasets_cfg import (
    make_object_dataset,
    make_scene_dataset,
)
from happypose.toolbox.datasets.pose_dataset import PoseDataset
from happypose.toolbox.datasets.scene_dataset import (
    IterableMultiSceneDataset,
    RandomIterableSceneDataset,
)
from happypose.toolbox.lib3d.rigid_mesh_database import MeshDataBase
from happypose.toolbox.renderer.panda3d_batch_renderer import Panda3dBatchRenderer
from happypose.toolbox.utils.logging import get_logger
from happypose.toolbox.utils.resources import (
    get_cuda_memory,
    get_gpu_memory,
    get_total_memory,
)

cudnn.benchmark = True
logger = get_logger(__name__)


class TestCosyposePoseTraining:
    @pytest.fixture(autouse=True)
    def setup(self):
        args = {"config": "ycbv-refiner-syntonly"}

        cfg_pose = OmegaConf.create({})

        logger.info(
            f"Training with config: {args['config']}",
        )
        if args["config"]:
            logger.info(
                f"Training with config: {args['config']}",
            )

        cfg_pose.resume_run_id = None

        N_CPUS = int(os.environ.get("N_CPUS", 10))
        N_WORKERS = min(N_CPUS - 2, 8)
        N_WORKERS = 8
        N_RAND = np.random.randint(1e6)

        run_comment = ""

        # Data
        cfg_pose.urdf_ds_name = "ycbv"
        cfg_pose.object_ds_name = "ycbv"
        cfg_pose.n_symmetries_batch = 64

        cfg_pose.train_ds_names = [
            ("synt.ycbv-1M", 1),
            ("ycbv.real.train", 3),
            ("ycbv.synthetic.train", 3),
        ]
        cfg_pose.val_ds_names = cfg_pose.train_ds_names
        cfg_pose.val_epoch_interval = 10
        cfg_pose.test_ds_names = ["ycbv.test.keyframes"]
        cfg_pose.test_epoch_interval = 30
        cfg_pose.n_test_frames = None

        cfg_pose.input_resize = (480, 640)
        cfg_pose.rgb_augmentation = True
        cfg_pose.background_augmentation = True
        cfg_pose.gray_augmentation = False

        # Model
        cfg_pose.backbone_str = "efficientnet-b3"
        cfg_pose.run_id_pretrain = None
        cfg_pose.n_pose_dims = 9
        cfg_pose.n_rendering_workers = N_WORKERS
        cfg_pose.refiner_run_id_for_test = None
        cfg_pose.coarse_run_id_for_test = "coarse-bop-ycbv-pbr--724183"

        # Optimizer
        cfg_pose.lr = 3e-4
        cfg_pose.weight_decay = 0.0
        cfg_pose.n_epochs_warmup = 50
        cfg_pose.lr_epoch_decay = 500
        cfg_pose.clip_grad_norm = 0.5

        # Training
        cfg_pose.batch_size = 16
        cfg_pose.epoch_size = 115200
        cfg_pose.n_epochs = 2
        cfg_pose.n_dataloader_workers = N_WORKERS

        # Method
        cfg_pose.loss_disentangled = True
        cfg_pose.n_points_loss = 2600
        cfg_pose.TCO_input_generator = "fixed"
        cfg_pose.n_iterations = 1
        cfg_pose.min_area = None

        if "bop-" in args["config"]:
            from happypose.pose_estimators.cosypose.cosypose.bop_config import (
                BOP_CONFIG,
                PBR_COARSE,
                PBR_REFINER,
            )

            bop_name, train_type, model_type = args["config"].split("-")[1:]
            bop_cfg = BOP_CONFIG[bop_name]
            if train_type == "pbr":
                cfg_pose.train_ds_names = [(bop_cfg["train_pbr_ds_name"][0], 1)]
            elif train_type == "synt+real":
                cfg_pose.train_ds_names = bop_cfg["train_synt_real_ds_names"]
                if model_type == "coarse":
                    PRETRAIN_MODELS = PBR_COARSE
                elif model_type == "refiner":
                    PRETRAIN_MODELS = PBR_REFINER
                cfg_pose.run_id_pretrain = PRETRAIN_MODELS[bop_name]
            else:
                raise ValueError

            cfg_pose.val_ds_names = cfg_pose.train_ds_names
            cfg_pose.urdf_ds_name = bop_cfg["urdf_ds_name"]
            cfg_pose.object_ds_name = bop_cfg["obj_ds_name"]
            cfg_pose.input_resize = bop_cfg["input_resize"]
            cfg_pose.test_ds_names = []

            if model_type == "coarse":
                cfg_pose.init_method = "z-up+auto-depth"
                cfg_pose.TCO_input_generator = "fixed+trans_noise"
                run_comment = "transnoise-zxyavg"
            elif model_type == "refiner":
                cfg_pose.TCO_input_generator = "gt+noise"
            else:
                raise ValueError

        elif "ycbv-" in args["config"]:
            cfg_pose.urdf_ds_name = "ycbv"
            cfg_pose.object_ds_name = "ycbv"
            cfg_pose.train_ds_names = [
                ("synthetic.ycbv-1M.train", 1),
                ("ycbv.train.synt", 1),
                ("ycbv.train.real", 3),
            ]
            cfg_pose.val_ds_names = [("synthetic.ycbv-1M.val", 1)]
            cfg_pose.test_ds_names = ["ycbv.test.keyframes"]
            cfg_pose.input_resize = (480, 640)

            if args["config"] == "ycbv-refiner-syntonly":
                cfg_pose.TCO_input_generator = "gt+noise"
                cfg_pose.train_ds_names = [("synthetic.ycbv-1M.train", 1)]
            elif args["config"] == "ycbv-refiner-finetune":
                cfg_pose.TCO_input_generator = "gt+noise"
                cfg_pose.run_id_pretrain = "ycbv-refiner-syntonly--596719"
            else:
                raise ValueError(args["config"])

        elif "tless-" in args["config"]:
            cfg_pose.urdf_ds_name = "tless.cad"
            cfg_pose.object_ds_name = "tless.cad"
            cfg_pose.train_ds_names = [
                ("synthetic.tless-1M.train", 1),
                ("tless.primesense.train", 5),
            ]
            cfg_pose.val_ds_names = [("synthetic.tless-1M.val", 1)]
            cfg_pose.test_ds_names = ["tless.primesense.test"]
            cfg_pose.input_resize = (540, 720)

            if args["config"] == "tless-coarse":
                cfg_pose.TCO_input_generator = "fixed"
            elif args["config"] == "tless-refiner":
                cfg_pose.TCO_input_generator = "gt+noise"

            # Ablations
            elif args["config"] == "tless-coarse-ablation-loss":
                cfg_pose.loss_disentangled = False
                cfg_pose.TCO_input_generator = "fixed"
            elif args["config"] == "tless-refiner-ablation-loss":
                cfg_pose.loss_disentangled = False
                cfg_pose.TCO_input_generator = "gt+noise"

            elif args["config"] == "tless-coarse-ablation-network":
                cfg_pose.TCO_input_generator = "fixed"
                cfg_pose.backbone_str = "flownet"
            elif args["config"] == "tless-refiner-ablation-network":
                cfg_pose.TCO_input_generator = "gt+noise"
                cfg_pose.backbone_str = "flownet"

            elif args["config"] == "tless-coarse-ablation-rot":
                cfg_pose.n_pose_dims = 7
                cfg_pose.TCO_input_generator = "fixed"
            elif args["config"] == "tless-refiner-ablation-rot":
                cfg_pose.n_pose_dims = 7
                cfg_pose.TCO_input_generator = "gt+noise"

            elif args["config"] == "tless-coarse-ablation-augm":
                cfg_pose.TCO_input_generator = "fixed"
                cfg_pose.rgb_augmentation = False
            elif args["config"] == "tless-refiner-ablation-augm":
                cfg_pose.TCO_input_generator = "gt+noise"
                cfg_pose.rgb_augmentation = False

            else:
                raise ValueError(args["config"])

        else:
            raise ValueError(args["config"])

        cfg_pose.run_id = f"{args['config']}-{run_comment}-{N_RAND}"

        N_GPUS = int(os.environ.get("N_PROCS", 1))
        cfg_pose.epoch_size = cfg_pose.epoch_size // N_GPUS
        self.cfg_pose = cfg_pose

    @pytest.mark.skip(
        reason="Currently, run two training tests (i.e. detector and pose) consecutively doesn't work with torch distributed"
    )
    def test_pose_training(self):
        if torch.cuda.is_available():
            train_pose(self.cfg_pose)
        else:
            pytest.skip("Training is not tested without GPU")


def train_pose(args):
    torch.set_num_threads(1)

    if args.resume_run_id:
        resume_dir = EXP_DIR / args.resume_run_id
        resume_args = yaml.load((resume_dir / "config.yaml").read_text())
        keep_fields = {"resume_run_id", "epoch_size"}
        vars(args).update(
            {k: v for k, v in vars(resume_args).items() if k not in keep_fields},
        )

    args.train_refiner = args.TCO_input_generator == "gt+noise"
    args.train_coarse = not args.train_refiner
    args.save_dir = EXP_DIR / args.run_id
    args = check_update_config(args)

    logger.info(f"{'-' * 80}")
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    logger.info(f"{'-' * 80}")

    # Initialize distributed
    device = torch.cuda.current_device()
    # seems to cause a problem in the tests
    init_distributed_mode()
    world_size = get_world_size()
    this_rank_epoch_size = args.epoch_size // get_world_size()
    this_rank_n_batch_per_epoch = this_rank_epoch_size // args.batch_size
    args.n_gpus = world_size
    args.global_batch_size = world_size * args.batch_size
    logger.info(f"Connection established with {world_size} gpus.")

    # Make train/val datasets
    def make_datasets(dataset_names):
        datasets = []
        deterministic = False
        for ds_name, n_repeat in dataset_names:
            assert "test" not in ds_name
            print("ds_name = ", ds_name)
            ds = make_scene_dataset(ds_name)
            iterator = RandomIterableSceneDataset(ds, deterministic=deterministic)
            logger.info(f"Loaded {ds_name} with {len(ds)} images.")
            for _ in range(n_repeat):
                datasets.append(iterator)
        return IterableMultiSceneDataset(datasets)

    scene_ds_train = make_datasets(args.train_ds_names)
    scene_ds_val = make_datasets(args.val_ds_names)

    ds_kwargs = {
        "resize": args.input_resize,
        "apply_rgb_augmentation": args.rgb_augmentation,
        "apply_background_augmentation": args.background_augmentation,
        "min_area": args.min_area,
        "apply_depth_augmentation": False,
    }
    ds_train = PoseDataset(scene_ds_train, **ds_kwargs)
    ds_val = PoseDataset(scene_ds_val, **ds_kwargs)

    # train_sampler = PartialSampler(ds_train, epoch_size=args.epoch_size)
    ds_iter_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        num_workers=args.n_dataloader_workers,
        collate_fn=ds_train.collate_fn,
        drop_last=False,
        pin_memory=True,
    )
    ds_iter_train = iter(ds_iter_train)

    # val_sampler = PartialSampler(ds_val, epoch_size=int(0.1 * args.epoch_size))
    ds_iter_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        num_workers=args.n_dataloader_workers,
        collate_fn=ds_val.collate_fn,
        drop_last=False,
        pin_memory=True,
    )
    ds_iter_val = iter(ds_iter_val)

    # Make model
    object_ds = make_object_dataset(args.object_ds_name)
    renderer = Panda3dBatchRenderer(
        object_ds,
        n_workers=args.n_rendering_workers,
        preload_cache=False,
    )
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    mesh_db_batched = mesh_db.batched(n_sym=args.n_symmetries_batch).cuda().float()

    model = create_pose_model_cosypose(
        cfg=args, renderer=renderer, mesh_db=mesh_db_batched
    ).cuda()

    eval_bundle = make_eval_bundle(args, model, mesh_db)

    if args.resume_run_id:
        resume_dir = EXP_DIR / args.resume_run_id
        path = resume_dir / "checkpoint.pth.tar"
        logger.info(f"Loading checkpoing from {path}")
        save = torch.load(path)
        state_dict = save["state_dict"]
        model.load_state_dict(state_dict)
        start_epoch = save["epoch"] + 1
    else:
        start_epoch = 0
    end_epoch = args.n_epochs

    if args.run_id_pretrain is not None:
        pretrain_path = EXP_DIR / args.run_id_pretrain / "checkpoint.pth.tar"
        logger.info(f"Using pretrained model from {pretrain_path}.")
        model.load_state_dict(torch.load(pretrain_path)["state_dict"])

    # Synchronize models across processes.
    model = sync_model(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device],
        output_device=device,
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Warmup
    if args.n_epochs_warmup == 0:

        def lambd(epoch):
            return 1

    else:
        n_batches_warmup = args.n_epochs_warmup * (args.epoch_size // args.batch_size)

        def lambd(batch):
            return (batch + 1) / n_batches_warmup

    lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambd)
    lr_scheduler_warmup.last_epoch = start_epoch * args.epoch_size // args.batch_size

    # LR schedulers
    # Divide LR by 10 every args.lr_epoch_decay
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_epoch_decay,
        gamma=0.1,
    )
    lr_scheduler.last_epoch = start_epoch - 1
    lr_scheduler.step()

    for epoch in range(start_epoch, end_epoch):
        meters_train = defaultdict(lambda: AverageValueMeter())
        meters_val = defaultdict(lambda: AverageValueMeter())
        meters_time = defaultdict(lambda: AverageValueMeter())

        h = functools.partial(
            h_pose,
            model=model,
            cfg=args,
            n_iterations=args.n_iterations,
            mesh_db=mesh_db_batched,
            input_generator=args.TCO_input_generator,
        )

        def train_epoch():
            model.train()
            iterator = tqdm(ds_iter_train, ncols=80)
            t = time.time()
            for n, data in enumerate(iterator):
                if n < 3:
                    if n > 0:
                        meters_time["data"].add(time.time() - t)

                    optimizer.zero_grad()

                    t = time.time()
                    loss = h(data=data, meters=meters_train)
                    meters_time["forward"].add(time.time() - t)
                    iterator.set_postfix(loss=loss.item())
                    meters_train["loss_total"].add(loss.item())

                    t = time.time()
                    loss.backward()
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=args.clip_grad_norm,
                        norm_type=2,
                    )
                    meters_train["grad_norm"].add(
                        torch.as_tensor(total_grad_norm).item()
                    )

                    optimizer.step()
                    meters_time["backward"].add(time.time() - t)
                    meters_time["memory"].add(
                        torch.cuda.max_memory_allocated() / 1024.0**2,
                    )

                    if epoch < args.n_epochs_warmup:
                        lr_scheduler_warmup.step()
                    t = time.time()
                else:
                    break
            if epoch >= args.n_epochs_warmup:
                lr_scheduler.step()

        @torch.no_grad()
        def validation():
            model.eval()
            for n, sample in enumerate(tqdm(ds_iter_val, ncols=80)):
                if n < 3:
                    loss = h(data=sample, meters=meters_val)
                    meters_val["loss_total"].add(loss.item())
                else:
                    break

        @torch.no_grad()
        def test():
            model.eval()
            return run_eval(eval_bundle, epoch=epoch)

        train_epoch()
        print("end train, stepping in to valida")
        if epoch % args.val_epoch_interval == 0:
            validation()

        # test_dict = None
        # if epoch % args.test_epoch_interval == 0:
        #    test_dict = test()

        print("updating dic")
        log_dict = {}
        log_dict.update(
            {
                "grad_norm": meters_train["grad_norm"].mean,
                "grad_norm_std": meters_train["grad_norm"].std,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "time_forward": meters_train["time_forward"].mean,
                "time_backward": meters_train["time_backward"].mean,
                "time_data": meters_train["time_data"].mean,
                "cuda_memory": get_cuda_memory(),
                "gpu_memory": get_gpu_memory(),
                "cpu_memory": get_total_memory(),
                "time": time.time(),
                "n_iterations": epoch * args.epoch_size // args.batch_size,
                "n_datas": epoch * this_rank_n_batch_per_epoch * args.batch_size,
            },
        )

        print("meters")
        for string, meters in zip(("train", "val"), (meters_train, meters_val)):
            for k in dict(meters).keys():
                log_dict[f"{string}_{k}"] = meters[k].mean

        print("reduce dict")
        log_dict = reduce_dict(log_dict)

        print("waiting on barrier")
        dist.barrier()
