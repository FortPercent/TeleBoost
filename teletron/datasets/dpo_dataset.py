# Example usage:
# from teletron.datasets.dpo_dataset import WanDPODataset
# dataset = WanDPODataset(
#     transforms=[],
#     dataset_base_path="/data",
#     dataset_metadata_path="/data/pairs.csv",
#     chosen_video_key="chosen",
#     rejected_video_key="rejected",
#     height=480,
#     width=832,
#     num_frames=49,
# )
# sample = dataset[0]
import torch, torchvision, imageio, os, json, pandas
import torch.distributed as dist
import numpy as np
import imageio.v3 as iio
from PIL import Image
from einops import repeat
from teleai_data_tool.file.file_client import FileClient
from teleai_data_tool.file.lmdb_client import LmdbClient
from my_utils import get_global_logger
from megatron.core import mpu


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)



class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)



class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data



class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)



class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)



class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)



class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image



## 输入是image PIL对象，输出是crop并resize后的PIL对象
class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image



class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    



class LoadVideoWithFileClient(DataProcessingOperator):
    def __init__(self, data_format="file"):
        """
        data_format: "file" or "lmdb"
        """
        self.data_format = data_format
        self.file_client = FileClient()
        self.lmdb_client = LmdbClient()

    def __call__(self, path: str):
        if self.data_format == "lmdb":
            return self.lmdb_client.get(path)
        else:
            return self.file_client.get(path)


class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        reader = imageio.get_reader(data)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames



class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]



class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames
    


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")



class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")



class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)



class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)



class PreprocessImageToTensor(DataProcessingOperator):
    def __init__(self, torch_dtype=torch.float32, device="cpu", pattern="B C H W", min_value=-1, max_value=1):
        self.torch_dtype = torch_dtype
        self.device = device
        self.pattern = pattern
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, image: Image.Image):
        tensor = torch.Tensor(np.array(image, dtype=np.float32))
        tensor = tensor.to(dtype=self.torch_dtype or tensor.dtype, device=self.device or tensor.device)
        tensor = tensor * ((self.max_value - self.min_value) / 255) + self.min_value
        tensor = repeat(tensor, f"H W C -> {self.pattern}", **({"B": 1} if "B" in self.pattern else {}))
        return tensor


class PreprocessVideoToTensor(DataProcessingOperator):
    def __init__(self, torch_dtype=torch.float32, device="cpu", pattern="B C T H W", min_value=-1, max_value=1):
        self.torch_dtype = torch_dtype
        self.device = device
        self.pattern = pattern
        self.min_value = min_value
        self.max_value = max_value
        parts = pattern.split()
        image_pattern = " ".join([p for p in parts if p != "T"])
        self.image_preprocess = PreprocessImageToTensor(
            torch_dtype=torch_dtype,
            device=device,
            pattern=image_pattern,
            min_value=min_value,
            max_value=max_value,
        )

    def __call__(self, video):
        if isinstance(video, Image.Image):
            video = [video]
        frames = [self.image_preprocess(image) for image in video]
        t_dim = self.pattern.index("T") // 2
        return torch.stack(frames, dim=t_dim)
    



class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        data_path_list=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        pipeline = None,
    ):
        self.base_path = base_path
        metadata_path = data_path_list if data_path_list is not None else metadata_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        from teletron.datasets.base_dataset import Compose
        self.pipeline = Compose(pipeline) if pipeline is not None else None
        self.load_metadata(metadata_path)
        try:
            from my_utils import UniversalDumper, DumpConfig, get_dumper
            self.dumper = get_dumper(DumpConfig(
                root_dir="./dump_dataset",
                enable=True,
                image_format="png",
                max_list_elems=64,
                max_dict_items=128,
                save_tensor_cpu=False,
            )
            )
        except Exception as e:
            print(f"UnifiedDataset: failed to create dumper: {e}")
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])

    @staticmethod
    def default_video_tensor_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        torch_dtype=torch.bfloat16, device="cpu", pattern="B C T H W",
        min_value=-1, max_value=1,
    ):
        return UnifiedDataset.default_video_operator(
            base_path=base_path,
            max_pixels=max_pixels,
            height=height,
            width=width,
            height_division_factor=height_division_factor,
            width_division_factor=width_division_factor,
            num_frames=num_frames,
            time_division_factor=time_division_factor,
            time_division_remainder=time_division_remainder,
        ) 
    
    
        #  >> PreprocessVideoToTensor(
        #     torch_dtype=torch_dtype,
        #     device=device,
        #     pattern=pattern,
        #     min_value=min_value,
        #     max_value=max_value,
        # )
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def inject_video_meta(self, data_dict):
        video = data_dict["video"]

        if "video_length" not in data_dict:
            if torch.is_tensor(video):
                if video.dim() >= 5:
                    video_length = int(video.shape[-3])
                elif video.dim() == 4:
                    if video.shape[0] in (1, 3, 4) and video.shape[1] not in (1, 3, 4):
                        video_length = int(video.shape[1])
                    elif video.shape[1] in (1, 3, 4) and video.shape[0] not in (1, 3, 4):
                        video_length = int(video.shape[0])
                    else:
                        video_length = int(video.shape[-3])
                else:
                    video_length = int(video.shape[0])
            elif isinstance(video, (list, tuple)):
                video_length = len(video)
            else:
                video_length = len(video)
            data_dict["video_length"] = video_length

        if "frame_interval" not in data_dict:
            data_dict["frame_interval"] = 1

        if "video_valid_range" not in data_dict:
            data_dict["video_valid_range"] = (0, data_dict["video_length"])

        if "video_height" not in data_dict or "video_width" not in data_dict:
            if torch.is_tensor(video):
                height = int(video.shape[-2])
                width = int(video.shape[-1])
            elif isinstance(video, (list, tuple)) and len(video) > 0:
                first = video[0]
                if torch.is_tensor(first):
                    height = int(first.shape[-2])
                    width = int(first.shape[-1])
                elif hasattr(first, "size"):
                    width, height = first.size
                else:
                    height = getattr(first, "height", 0)
                    width = getattr(first, "width", 0)
            else:
                try:
                    height = video.height
                    width = video.width
                except Exception:
                    frame0 = video.get_frames_at([0]).data
                    height = frame0.shape[-2]
                    width = frame0.shape[-1]
            data_dict["video_height"] = int(height)
            data_dict["video_width"] = int(width)

        from teletron.utils import set_config
        ds_config = set_config().get("dataset", {})

        data_dict["video_info"] = (ds_config["width"], ds_config["height"])

        data_dict.setdefault("struct_prompt", "")
        data_dict.setdefault("short_prompt", "")
        data_dict.setdefault("dense_prompt", "")
        return data_dict
    def _get_dp_rank(self):
        env_local = os.environ.get("LOCAL_RANK")
        if env_local is not None:
            try:
                return int(env_local)
            except ValueError:
                pass
        try:
            return mpu.get_data_parallel_rank(with_context_parallel=True)
        except TypeError:
            return mpu.get_data_parallel_rank()
        except Exception:
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank()
            env_rank = os.environ.get("RANK")
            return int(env_rank) if env_rank is not None else 0

    def _resolve_raw_dataset_path(self):
        raw_path = os.environ.get("WAN_DPO_DATASET_RAW_FILE") or os.environ.get("WAN_DPO_DATASET_DUMP_FILE")
        if not raw_path:
            return None
        rank = self._get_dp_rank()
        if "{rank}" in raw_path:
            return raw_path.format(rank=rank)
        if os.path.exists(raw_path):
            return raw_path
        root, ext = os.path.splitext(raw_path)
        candidate = f"{root}_rank{rank}{ext}"
        if os.path.exists(candidate):
            return candidate
        return raw_path

    def _load_raw_dataset(self, path):
        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if record.get("tag") != "dataset.raw":
                    continue
                stage = record.get("stage")
                if stage is not None and stage != "raw":
                    continue
                payload = record.get("payload")
                if not isinstance(payload, dict):
                    continue
                payload = payload.copy()
                dump_id = record.get("dump_id")
                data_id = record.get("data_id")
                if "dpo_pair_id" not in payload:
                    if data_id is not None:
                        payload["dpo_pair_id"] = data_id
                    elif dump_id is not None:
                        payload["dpo_pair_id"] = dump_id
                records.append((dump_id, payload))
        records.sort(key=lambda x: x[0] if x[0] is not None else -1)
        return [payload for _, payload in records]



    def load_metadata(self, metadata_path):
        if os.environ.get("WAN_DPO_PREVAE_COMPARE", "0") == "1":
            raw_path = self._resolve_raw_dataset_path()
            if raw_path:
                self.data = self._load_raw_dataset(raw_path)
                logger = self._get_logger()
                logger.info(f"[UnifiedDataset] loaded raw dump: {raw_path} count={len(self.data)}")
                return
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
            return

        def _load_single_metadata(path):
            if isinstance(path, list):
                return list(path)
            if isinstance(path, dict):
                return [path]
            if not isinstance(path, str):
                raise TypeError(f"metadata_path must be str or list, got {type(path)}")
            if path.endswith(".json"):
                with open(path, "r") as f:
                    metadata = json.load(f)
                return metadata if isinstance(metadata, list) else [metadata]
            if path.endswith(".jsonl"):
                metadata = []
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        metadata.append(json.loads(line))
                return metadata
            metadata = pandas.read_csv(path)
            return [metadata.iloc[i].to_dict() for i in range(len(metadata))]

        if isinstance(metadata_path, (list, tuple)):
            combined = []
            for path in metadata_path:
                combined.extend(_load_single_metadata(path))
            self.data = combined
            return

        self.data = _load_single_metadata(metadata_path)

    def _get_logger(self):
        return get_global_logger()

    def _get_producer_rank(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        env_rank = os.environ.get("RANK")
        return int(env_rank) if env_rank is not None else -1

    def _log_data(self, data_id, data, stage):
        logger = self._get_logger()
        producer_rank = self._get_producer_rank()
        data_len = len(self.cached_data) if self.load_from_cache else len(self.data)
        sample_idx = data_id % data_len if data_len > 0 else -1
        source = "cache" if self.load_from_cache else "metadata"
        data_keys = sorted(list(data.keys())) if isinstance(data, dict) else []
        logger.info(
            f"[UnifiedDataset __getitem__] rank={producer_rank} source={source} "
            f"data_len={data_len} data_id={data_id} sample_idx={sample_idx} "
            f"stage={stage} keys={data_keys}"
        )

    def _log_data_dump(self, data_id, data, stage, branch=None):
        # branch 你可以传 chosen / rejected 或 pipeline 名称
        if self._get_dp_rank() == 0:
            self.dumper.dump(stage=stage, obj=data, data_id=data_id, branch=branch)

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
            self._log_data(data_id, data, "processed")
            return data

        def _fmt_keys(d):
            return sorted(list(d.keys()))

        # ===== 0️⃣ 取原始样本（完整 dict）=====
        raw = self.data[data_id % len(self.data)].copy()
        # print(f"[UnifiedDataset __getitem__] raw keys: {_fmt_keys(raw)}")
        # self._log_data_dump(data_id, raw, "raw_data")
        # ===== 1️⃣ path → VideoDecoder（只处理 video key）=====
        for key in self.data_file_keys:
            if key not in raw:
                continue

            if key in self.special_operator_map:
                raw[key] = self.special_operator_map[key](raw[key])
            else:
                # 现在这里会得到chosen和rejected的PIL列表
                raw[key] = self.main_data_operator(raw[key])
        # print(f"[UnifiedDataset __getitem__] after decode video keys: {[k for k in self.data_file_keys if k in raw]}")
        # self._log_data_dump(data_id, raw, "after_video_decode")
        # ===== 2️⃣ 明确“公共字段”=====
        # 除了 chosen / rejected，其它都视为公共字段，prompt
        shared_fields = {
            k: v for k, v in raw.items()
            if k not in self.data_file_keys
        }
        # print(f"[UnifiedDataset __getitem__] shared fields: {_fmt_keys(shared_fields)}")

        out = {}

        # ===== 3️⃣ 分别构建 chosen / rejected 分支 =====
        for key in self.data_file_keys:
            if key not in raw:
                continue

            # ---- 核心：公共字段 + 私有 video ----
            data_i = shared_fields.copy()
            data_i["video"] = raw[key]
            pair_id = raw.get("dpo_pair_id", data_id)
            data_i["dpo_pair_id"] = int(pair_id)
            data_i["dpo_branch"] = key

            # 注入 video meta（video_length / height / width / fps 等）
            data_i = self.inject_video_meta(data_i)

            # 跑 dict-level pipeline
            if self.pipeline is not None:
                # print(f"before pp = {data_i.keys()}")
                # self._log_data(data_id, data_i, f"before_pipeline_{key}")
                # self._log_data_dump(data_id, data_i, f"before_pipeline", branch=key)
                data_i = self.pipeline(data_i)
                # print(f"after pp = {data_i.keys()}")
                # self._log_data_dump(data_id, data_i, f"after_pipeline", branch=key)
            for k, v in shared_fields.items():
                data_i.setdefault(k, v)
            out[key] = data_i
            # print(f"[UnifiedDataset __getitem__] branch '{key}' keys: {_fmt_keys(data_i)}")

        # print(f"[UnifiedDataset __getitem__] output branches: {_fmt_keys(out)}")

        # self._log_data_dump(data_id, out, "processed")
        return out

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True






class WanDPODataset(UnifiedDataset):
    def __init__(
        self,
        # === 对应 args / config ===
        transforms,
        dataset_base_path="",
        dataset_metadata_path=None,
        data_path_list=None,
        dataset_repeat=1,

        chosen_video_key="chosen",
        rejected_video_key="rejected",

        height=None,
        width=None,
        num_frames=49,
        max_pixels=1920 * 1080,

        time_division_factor=4,
        time_division_remainder=1,

        height_division_factor=16,
        width_division_factor=16,

        **kwargs,
    ):
        data_file_keys = (chosen_video_key, rejected_video_key)

        main_data_operator = UnifiedDataset.default_video_tensor_operator(
            base_path=dataset_base_path,
            max_pixels=max_pixels,
            height=height,
            width=width,
            height_division_factor=height_division_factor,
            width_division_factor=width_division_factor,
            num_frames=num_frames,
            time_division_factor=time_division_factor,
            time_division_remainder=time_division_remainder,
            torch_dtype=torch.bfloat16,
            device="cpu",
            pattern="B C T H W",
            min_value=-1,
            max_value=1,
        )

        
        # UnifiedDataset.default_video_operator(
        #     base_path=dataset_base_path,
        #     max_pixels=max_pixels,
        #     height=height,
        #     width=width,
        #     height_division_factor=height_division_factor,
        #     width_division_factor=width_division_factor,
        #     num_frames=num_frames,
        #     time_division_factor=time_division_factor,
        #     time_division_remainder=time_division_remainder,
        # )

        # 可选：保持和原脚本一致
        special_operator_map = {
            "animate_face_video":
                ToAbsolutePath(dataset_base_path)
                >> LoadVideo(
                    num_frames,
                    time_division_factor,
                    time_division_remainder,
                    frame_processor=ImageCropAndResize(
                        512, 512, None, 16, 16
                    )
                )
                >> LoadVideoWithFileClient(data_format="file"),
        }

        super().__init__(
            base_path=dataset_base_path,
            metadata_path=dataset_metadata_path,
            data_path_list=data_path_list,
            repeat=dataset_repeat,
            data_file_keys=data_file_keys,
            main_data_operator=main_data_operator,
            special_operator_map=special_operator_map,
            pipeline = transforms
        )
