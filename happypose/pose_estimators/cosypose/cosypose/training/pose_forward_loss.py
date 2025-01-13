import numpy as np
import torch

from happypose.pose_estimators.cosypose.cosypose.lib3d.cosypose_ops import (
    TCO_init_from_boxes,
    TCO_init_from_boxes_zup_autodepth,
    loss_refiner_CO_disentangled,
    loss_refiner_CO_disentangled_quaternions,
)
from happypose.pose_estimators.cosypose.cosypose.lib3d.mesh_losses import (
    compute_ADD_L1_loss,
)
from happypose.pose_estimators.cosypose.cosypose.lib3d.transform_ops import add_noise


def cast(obj):
    return obj.cuda(non_blocking=True)


def h_pose(model, mesh_db, data, meters, cfg, n_iterations=1, input_generator="fixed"):
    batch_size, _, h, w = data.rgbs.shape

    images = cast(data.rgbs).float() / 255.0
    K = cast(data.K).float()
    TCO_gt = cast(data.TCO).float()
    labels = np.array([obj.label for obj in data.object_datas])
    bboxes = cast(data.bboxes).float()

    meshes = mesh_db.select(labels)
    points = meshes.sample_points(cfg.n_points_loss, deterministic=False)
    TCO_possible_gt = TCO_gt.unsqueeze(1) @ meshes.symmetries

    if input_generator == "fixed":
        TCO_init = TCO_init_from_boxes(z_range=(1.0, 1.0), boxes=bboxes, K=K)
    elif input_generator == "gt+noise":
        TCO_init = add_noise(
            TCO_possible_gt[:, 0],
            euler_deg_std=[15, 15, 15],
            trans_std=[0.01, 0.01, 0.05],
        )
    elif input_generator == "fixed+trans_noise":
        assert cfg.init_method == "z-up+auto-depth"
        TCO_init = TCO_init_from_boxes_zup_autodepth(bboxes, points, K)
        TCO_init = add_noise(
            TCO_init,
            euler_deg_std=[0, 0, 0],
            trans_std=[0.01, 0.01, 0.05],
        )
    else:
        msg = "Unknown input generator"
        raise ValueError(msg, input_generator)

    # model.module.enable_debug()
    outputs = model(
        images=images,
        K=K,
        labels=labels,
        TCO=TCO_init,
        n_iterations=n_iterations,
    )
    # raise ValueError

    losses_TCO_iter = []
    for n in range(n_iterations):
        iter_outputs = outputs[f"iteration={n + 1}"]
        K_crop = iter_outputs.K_crop
        TCO_input = iter_outputs.TCO_input
        TCO_pred = iter_outputs.TCO_output
        model_outputs = iter_outputs.model_outputs

        if cfg.loss_disentangled:
            if cfg.n_pose_dims == 9:
                loss_fn = loss_refiner_CO_disentangled
            elif cfg.n_pose_dims == 7:
                loss_fn = loss_refiner_CO_disentangled_quaternions
            else:
                raise ValueError
            pose_outputs = model_outputs["pose"]
            loss_TCO_iter = loss_fn(
                TCO_possible_gt=TCO_possible_gt,
                TCO_input=TCO_input,
                refiner_outputs=pose_outputs,
                K_crop=K_crop,
                points=points,
            )
        else:
            loss_TCO_iter = compute_ADD_L1_loss(
                TCO_possible_gt[:, 0],
                TCO_pred,
                points,
            )

        meters[f"loss_TCO-iter={n + 1}"].add(loss_TCO_iter.mean().item())
        losses_TCO_iter.append(loss_TCO_iter)

    loss_TCO = torch.cat(losses_TCO_iter).mean()
    loss = loss_TCO
    meters["loss_TCO"].add(loss_TCO.item())
    meters["loss_total"].add(loss.item())
    return loss
