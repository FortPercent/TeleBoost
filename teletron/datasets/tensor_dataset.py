import torch
import pandas as pd
import os

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, pth_paths, metadata_paths, *args):
        # Ensure inputs are lists
        if isinstance(pth_paths, str):
            pth_paths = [pth_paths]
        if isinstance(metadata_paths, str):
            metadata_paths = [metadata_paths]
        # assert len(pth_paths) == len(metadata_paths), "Mismatch between pth_paths and metadata_paths"

        self.path = []

        # Load each dataset source
        for pth_path, metadata_path in zip(pth_paths, metadata_paths):
            metadata = pd.read_csv(metadata_path)
            print(f"🔍 {len(metadata)} entries found in {metadata_path}")
            
            if "file_name" in metadata.columns:
                name_column = "file_name"
            elif "file_path" in metadata.columns:
                name_column = "file_path"

            # Construct full tensor paths and check for file existence
            for file_name in metadata[name_column]:
                tensor_path = os.path.join(pth_path, file_name) + ".tensors.pth"
                self.path.append(tensor_path)

        print(f"✅ Total valid tensor files loaded: {len(self.path)}")
        assert len(self.path) > 0, "No valid tensor files found."

    def __getitem__(self, index):
        # Generate a pseudo-random offset for this index (helps randomize sample order)
        # data_id = torch.randint(0, len(self.path), (1,))[0]
        # data_id = (data_id + index) % len(self.path)
        # path = self.path[data_id]

        # Load tensor from file
        path = self.path[index]
        data = torch.load(path, weights_only=True, map_location="cpu")
        return data

    def __len__(self):
        # Total number of available tensor files
        return len(self.path)
