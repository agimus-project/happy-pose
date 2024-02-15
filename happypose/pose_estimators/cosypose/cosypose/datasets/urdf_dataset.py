from pathlib import Path

import pandas as pd


class UrdfDataset:
    def __init__(self, ds_dir):
        ds_dir = Path(ds_dir)
        index = []
        for urdf_dir in Path(ds_dir).iterdir():
            urdf_paths = list(urdf_dir.glob("*.urdf"))
            if len(urdf_paths) == 1:
                urdf_path = urdf_paths[0]
                infos = {
                    "label": urdf_dir.name,
                    "urdf_path": urdf_path.as_posix(),
                    "scale": 1.0,
                }
                index.append(infos)
        self.index = pd.DataFrame(index)

    def __getitem__(self, idx):
        return self.index.iloc[idx]

    def __len__(self):
        return len(self.index)


class OneUrdfDataset:
    def __init__(self, urdf_path, label, scale=1.0):
        index = [
            {"urdf_path": urdf_path, "label": label, "scale": scale},
        ]
        self.index = pd.DataFrame(index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.index.iloc[idx]
