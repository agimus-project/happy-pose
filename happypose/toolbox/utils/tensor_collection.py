"""Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Standard Library
from pathlib import Path
from typing import List

# Third Party
import pandas as pd
import torch

# MegaPose
from happypose.toolbox.utils.distributed import get_rank, get_world_size


def concatenate(datas):
    datas = [data for data in datas if len(data) > 0]
    if len(datas) == 0:
        return PandasTensorCollection(infos=pd.DataFrame())
    classes = [data.__class__ for data in datas]
    assert all(class_n == classes[0] for class_n in classes)

    infos = pd.concat([data.infos for data in datas], axis=0, sort=False).reset_index(
        drop=True,
    )
    tensor_keys = datas[0].tensors.keys()
    tensors = {}
    for k in tensor_keys:
        tensors[k] = torch.cat([getattr(data, k) for data in datas], dim=0)
    return PandasTensorCollection(infos=infos, **tensors)


class TensorCollection:
    def __init__(self, **kwargs):
        self.__dict__["_tensors"] = {}
        for k, v in kwargs.items():
            self.register_tensor(k, v)

    def register_tensor(self, name, tensor):
        self._tensors[name] = tensor

    def delete_tensor(self, name):
        del self._tensors[name]

    def __repr__(self):
        s = self.__class__.__name__ + "(\n"
        for k, t in self._tensors.items():
            s += f"    {k}: {t.shape} {t.dtype} {t.device},\n"
        s += ")"
        return s

    def __getitem__(self, ids):
        tensors = {}
        for k, _v in self._tensors.items():
            tensors[k] = getattr(self, k)[ids]
        return TensorCollection(**tensors)

    def __getattr__(self, name):
        if name in self._tensors:
            return self._tensors[name]
        elif name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError

    @property
    def tensors(self):
        return self._tensors

    @property
    def device(self):
        return next(iter(self.tensors.values())).device

    def __getstate__(self):
        return {"tensors": self.tensors}

    def __setstate__(self, state):
        self.__init__(**state["tensors"])
        return

    def __setattr__(self, name, value):
        if "_tensors" not in self.__dict__:
            msg = "Please call __init__"
            raise ValueError(msg)
        if name in self._tensors:
            self._tensors[name] = value
        else:
            self.__dict__[name] = value

    def to(self, torch_attr):
        for k, v in self._tensors.items():
            self._tensors[k] = v.to(torch_attr)
        return self

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")

    def float(self):
        return self.to(torch.float)

    def double(self):
        return self.to(torch.double)

    def half(self):
        return self.to(torch.half)

    def clone(self):
        tensors = {}
        for k, _v in self.tensors.items():
            tensors[k] = getattr(self, k).clone()
        return TensorCollection(**tensors)


class PandasTensorCollection(TensorCollection):
    def __init__(self, infos, **tensors):
        super().__init__(**tensors)
        self.infos = infos.reset_index(drop=True)
        self.meta = {}

    def register_buffer(self, k, v):
        assert len(v) == len(self)
        super().register_buffer()

    def merge_df(self, df, *args, **kwargs):
        infos = self.infos.merge(df, how="left", *args, **kwargs)
        assert len(infos) == len(self.infos)
        assert (infos.index == self.infos.index).all()
        return PandasTensorCollection(infos=infos, **self.tensors)

    def clone(self):
        tensors = super().clone().tensors
        return PandasTensorCollection(self.infos.copy(), **tensors)

    def __repr__(self):
        s = self.__class__.__name__ + "(\n"
        for k, t in self._tensors.items():
            s += f"    {k}: {t.shape} {t.dtype} {t.device},\n"
        s += f"{'-' * 40}\n"
        s += "    infos:\n" + self.infos.__repr__() + "\n"
        s += ")"
        return s

    def __getitem__(self, ids):
        infos = self.infos.iloc[ids].reset_index(drop=True)
        tensors = super().__getitem__(ids).tensors
        return PandasTensorCollection(infos, **tensors)

    def __len__(self):
        return len(self.infos)

    def gather_distributed(self, tmp_dir=None):
        rank, world_size = get_rank(), get_world_size()
        tmp_file_template = (tmp_dir / "rank={rank}.pth.tar").as_posix()

        if rank > 0:
            tmp_file = tmp_file_template.format(rank=rank)
            torch.save(self, tmp_file)

        if world_size > 1:
            torch.distributed.barrier()

        datas = [self]
        if rank == 0 and world_size > 1:
            for n in range(1, world_size):
                tmp_file = tmp_file_template.format(rank=n)
                data = torch.load(tmp_file)
                datas.append(data)
                Path(tmp_file).unlink()

        if world_size > 1:
            torch.distributed.barrier()
        return concatenate(datas)

    def __getstate__(self):
        state = super().__getstate__()
        state["infos"] = self.infos
        state["meta"] = self.meta
        return state

    def __setstate__(self, state):
        self.__init__(state["infos"], **state["tensors"])
        self.meta = state["meta"]
        return


def filter_top_pose_estimates(
    data_TCO: PandasTensorCollection,
    top_K: int,
    group_cols: List[str],
    filter_field: str,
    ascending: bool = False,
) -> PandasTensorCollection:
    """Filter the pose estimates by retaining only the top-K coarse model scores.

    Retain only the top_K estimates corresponding to each hypothesis_id

    Args:
        top_K: how many estimates to retain
        group_cols: group of columns among which sorting should be done
        filter_field: the field to filter estimates by
        ascending: should filter_field
    """

    df = data_TCO.infos

    # Logic from https://stackoverflow.com/a/40629420
    df = (
        df.sort_values(filter_field, ascending=ascending)
        .groupby(group_cols)
        .head(top_K)
    )

    data_TCO_filtered = data_TCO[df.index.tolist()]

    return data_TCO_filtered
