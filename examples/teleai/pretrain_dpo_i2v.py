import os
import torch
from teletron.train import Trainer, parse_args
import torch.distributed as dist
from megatron.core import mpu
from teletron.models.flow_match import FlowMatchScheduler
from teletron.train.utils import get_batch, loss_func as base_loss_func, average_losses_across_data_parallel_group
from teletron.utils import get_timers, set_config
import torch.nn.functional as F

def dpo_loss_func(output_tensor):
    print(f"[Rank {torch.distributed.get_rank()}] enter dpo_loss_func")
    # output_tensor 可能是 [loss_reject_scaled, loss_chosen_scaled, loss_reject, loss_chosen, dpo_loss]
    if not isinstance(output_tensor, (list, tuple)):
        output_tensor = [output_tensor]

    loss_for_backward = output_tensor[:2]
    # 这两个 loss 已经是标量（你 forward 里 .sum() 了），这里不需要 .mean()
    # 但为了安全，还是统一成 scalar
    losses = [t if t.dim() == 0 else t.mean() for t in loss_for_backward]

    # 返回给 deepspeed_forward_step 的 "loss" 应该是一个 Tensor（或 list），用于 backward
    # 我们让它保持 list（长度2），后面 backward_step 里循环两次 backward。
    loss_for_backward = losses

    # 日志：给一个总的 dpo_loss（只是显示用，不参与反传）
    loss_total = sum(losses).detach()
    if len(output_tensor) >= 5:
        dpo_loss = output_tensor[4]
        dpo_loss_mean = dpo_loss if dpo_loss.dim() == 0 else dpo_loss.mean()
    else:
        dpo_loss_mean = loss_total

    if len(output_tensor) >= 4:
        loss_reject = output_tensor[2]
        loss_chosen = output_tensor[3]
        loss_reject_mean = loss_reject if loss_reject.dim() == 0 else loss_reject.mean()
        loss_chosen_mean = loss_chosen if loss_chosen.dim() == 0 else loss_chosen.mean()
        averaged = average_losses_across_data_parallel_group(
            [dpo_loss_mean.detach(), loss_reject_mean.detach(), loss_chosen_mean.detach()]
        )
        loss_dict = {
            "loss": averaged[0],
            "loss_reject_mean": averaged[1],
            "loss_chosen_mean": averaged[2],
        }
    else:
        averaged = average_losses_across_data_parallel_group([dpo_loss_mean.detach()])
        loss_dict = {"loss": averaged[0]}
    print(f"[Rank {torch.distributed.get_rank()}] leave dpo_loss_func loss scaled = {loss_for_backward}")
    return loss_for_backward, loss_dict



def extra_args(parser):
    group = parser.add_argument_group(title='customized args')
    # follow this format to add
    # group.add_argument("--test_valid", type=str, default="")
    group.add_argument("--moe-step-factor-list", type=float, action='append')
    group.add_argument("--test-with-pseudo-data", action="store_true")
    group.add_argument("--test-resolution", type=str, default="360")
    group.add_argument("--save-dumps", action="store_true")
    group.add_argument("--save-dumps-dir", type=str, default=None)
    group.add_argument("--save-dumps-interval", type=int, default=1)
    group.add_argument("--use-saved-inputs", action="store_true")
    group.add_argument("--compare-saved-losses", action="store_true")
    group.add_argument("--compare-losses-rtol", type=float, default=1e-5)
    group.add_argument("--compare-losses-atol", type=float, default=1e-8)
    group.add_argument("--noise-seed", type=int, default=None)
    
    return parser

def _compute_single_loss(
    latents,
    context,
    clip_feature,
    y,
    flow_scheduler,
    model,
    timestep,
    noise,
    return_debug=False,
):
    training_target = flow_scheduler.training_target(latents, noise, timestep)
    noisy_latents = flow_scheduler.add_noise(latents, noise, timestep)
    loss_weight = flow_scheduler.training_weight(timestep)

  
    output_tensor_list = model(
        x=noisy_latents,
        timestep=timestep,
        context=context,
        clip_feature=clip_feature,
        y=y,
    )
    print(f"noisy_latents = {noisy_latents.shape},output_tensor_list = {output_tensor_list.shape} training_target = {training_target.shape}")
    loss = torch.nn.functional.mse_loss(
        output_tensor_list.float(), training_target.float()
    )
    loss_wo_w = loss
    loss = loss * loss_weight

    first_frame_loss = loss_wo_w.new_zeros(())

    if return_debug:
        debug = {
            "latents": latents,
            "noise": noise,
            "noisy_latents": noisy_latents,
            "training_target": training_target,
            "loss_weight": loss_weight,
            "noise_pred": output_tensor_list.detach(),
        }
        return loss, loss_wo_w, first_frame_loss, debug

    return loss, loss_wo_w, first_frame_loss


def _detach_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _detach_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_detach_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_to_cpu(v) for v in obj)
    return obj


def _to_cuda(obj, device=None):
    if torch.is_tensor(obj):
        return obj.to(device=device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_cuda(v, device=device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cuda(v, device=device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cuda(v, device=device) for v in obj)
    return obj


def _dtype_from_string(dtype_str: str):
    name = dtype_str.split(".", 1)[-1] if dtype_str.startswith("torch.") else dtype_str
    if not hasattr(torch, name):
        raise ValueError(f"Unsupported dtype string: {dtype_str}")
    return getattr(torch, name)


def _broadcast_tensor_from_src(tensor, src_rank, group, device):
    if tensor is None and torch.distributed.get_rank() == src_rank:
        meta = {"is_none": True}
    elif torch.distributed.get_rank() == src_rank:
        meta = {
            "is_none": False,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    else:
        meta = {"is_none": True}

    meta_list = [meta]
    dist.broadcast_object_list(meta_list, src=src_rank, group=group)
    meta = meta_list[0]
    if meta.get("is_none"):
        return None

    dtype = _dtype_from_string(meta["dtype"])
    if torch.distributed.get_rank() == src_rank:
        out = tensor.to(device=device, non_blocking=True)
    else:
        out = torch.empty(meta["shape"], dtype=dtype, device=device)

    dist.broadcast(out, src=src_rank, group=group)
    return out


def _set_nested(root, path, value):
    node = root
    for key in path[:-1]:
        if key not in node:
            node[key] = {}
        node = node[key]
    node[path[-1]] = value


def _broadcast_saved_payload(payload, device):
    cp_world = mpu.get_tensor_context_parallel_world_size()
    if cp_world <= 1:
        return _to_cuda(payload, device=device)

    src_rank = mpu.get_tensor_context_parallel_src_rank()
    group = mpu.get_tensor_context_parallel_group()
    tensor_paths = [
        ("timestep",),
        ("context",),
        ("chosen", "clip_feature"),
        ("chosen", "y"),
        ("chosen", "latents"),
        ("chosen", "noise"),
        ("chosen", "noisy_latents"),
        ("chosen", "noise_pred"),
        ("chosen", "training_target"),
        ("chosen", "loss_weight"),
        ("rejected", "clip_feature"),
        ("rejected", "y"),
        ("rejected", "latents"),
        ("rejected", "noise"),
        ("rejected", "noisy_latents"),
        ("rejected", "noise_pred"),
        ("rejected", "training_target"),
        ("rejected", "loss_weight"),
        ("losses", "loss_chosen"),
        ("losses", "loss_reject"),
        ("losses", "dpo_loss"),
        ("losses", "loss_wo_w_chosen"),
        ("losses", "loss_wo_w_reject"),
        ("losses", "first_frame_loss_chosen"),
        ("losses", "first_frame_loss_reject"),
    ]

    out = {}
    for path in tensor_paths:
        if payload is not None:
            node = payload
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
            tensor = node
        else:
            tensor = None
        broadcasted = _broadcast_tensor_from_src(tensor, src_rank, group, device)
        _set_nested(out, path, broadcasted)
    return out


def _compute_single_loss_from_saved(
    noisy_latents,
    training_target,
    loss_weight,
    context,
    clip_feature,
    y,
    model,
    timestep,
    expected_noise_pred=None,
    compare_name=None,
    rtol=1e-5,
    atol=1e-8,
):
    tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
    if dist.get_rank() == tp_cp_src_rank:
        inputs_info = {
            "x": (str(noisy_latents.dtype), list(noisy_latents.shape)),
            "timestep": (str(timestep.dtype), list(timestep.shape)),
            "context": (str(context.dtype), list(context.shape)),
            "clip_feature": (
                str(clip_feature.dtype) if torch.is_tensor(clip_feature) else str(type(clip_feature).__name__),
                list(clip_feature.shape) if torch.is_tensor(clip_feature) else None,
            ),
            "y": (
                str(y.dtype) if torch.is_tensor(y) else str(type(y).__name__),
                list(y.shape) if torch.is_tensor(y) else None,
            ),
        }
        print(f"[DPO input-dtype] rank={dist.get_rank()} {compare_name or 'noise_pred'} inputs={inputs_info}")
    output_tensor_list = model(
        x=noisy_latents,
        timestep=timestep,
        context=context,
        clip_feature=clip_feature,
        y=y,
    )
    if expected_noise_pred is not None:
        mismatch = _compare_input_tensor(
            compare_name or "noise_pred",
            expected_noise_pred,
            output_tensor_list,
            rtol=rtol,
            atol=atol,
        )
        if mismatch is not None:
            print(f"[DPO output-check] rank={dist.get_rank()} mismatch={mismatch}")
        elif dist.get_rank() == tp_cp_src_rank:
            print(f"[DPO output-check] rank={dist.get_rank()} {compare_name or 'noise_pred'} ok")
    loss = torch.nn.functional.mse_loss(
        output_tensor_list.float(), training_target.float()
    )
    loss_wo_w = loss
    loss = loss * loss_weight

    first_frame_loss = loss_wo_w.new_zeros(())

    return loss, loss_wo_w, first_frame_loss


def _compare_losses(saved_losses, current_losses, rtol, atol):
    results = {}
    for key, current in current_losses.items():
        saved = saved_losses.get(key)
        if saved is None:
            results[key] = {"missing": True}
            continue
        if not torch.is_tensor(saved):
            results[key] = {"missing": True}
            continue
        if saved.shape != current.shape:
            results[key] = {
                "shape_mismatch": True,
                "saved_shape": list(saved.shape),
                "current_shape": list(current.shape),
            }
            continue
        diff = (current - saved).abs()
        results[key] = {
            "missing": False,
            "allclose": bool(torch.allclose(current, saved, rtol=rtol, atol=atol)),
            "max_abs": diff.max().item(),
            "mean_abs": diff.mean().item(),
        }
    return results


def _compare_input_tensor(name, expected, actual, rtol=1e-5, atol=1e-8):
    if expected is None and actual is None:
        return None
    if expected is None or actual is None:
        return {"name": name, "reason": "one is None"}
    if not torch.is_tensor(expected) or not torch.is_tensor(actual):
        if expected == actual:
            return None
        return {"name": name, "reason": "non-tensor mismatch"}
    if expected.shape != actual.shape:
        return {
            "name": name,
            "reason": "shape mismatch",
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
        }
    if expected.dtype != actual.dtype:
        return {
            "name": name,
            "reason": "dtype mismatch",
            "expected_dtype": str(expected.dtype),
            "actual_dtype": str(actual.dtype),
        }
    if torch.equal(expected, actual):
        return None
    allclose = torch.allclose(expected, actual, rtol=rtol, atol=atol)
    if allclose:
        return None
    diff = (expected - actual).abs()
    return {
        "name": name,
        "reason": "value mismatch",
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
    }


def _check_payload_inputs(payload, inputs, rtol=1e-5, atol=1e-8):
    mismatches = []
    for name, pair in inputs.items():
        expected, actual = pair
        diff = _compare_input_tensor(name, expected, actual, rtol=rtol, atol=atol)
        if diff is not None:
            mismatches.append(diff)
    return mismatches


def _summarize_scheduler(scheduler, max_values=20):
    if scheduler is None:
        return {}
    summary = {"class": scheduler.__class__.__name__}
    sigmas = getattr(scheduler, "sigmas", None)
    if isinstance(sigmas, torch.Tensor):
        summary["num_inference_steps"] = int(sigmas.numel())
    for key, value in scheduler.__dict__.items():
        if key.startswith("_"):
            continue
        if torch.is_tensor(value):
            t = value.detach().cpu()
            entry = {"shape": list(t.shape), "dtype": str(t.dtype)}
            if t.numel() <= max_values:
                entry["values"] = [float(x) for x in t.flatten().tolist()]
            elif t.numel() > 0:
                flat = t.float().flatten()
                entry["stats"] = {
                    "min": float(flat.min()),
                    "max": float(flat.max()),
                    "mean": float(flat.mean()),
                    "std": float(flat.std()),
                }
                entry["preview"] = [float(x) for x in flat[:max_values].tolist()]
            summary[key] = entry
        elif isinstance(value, (int, float, bool)) or value is None:
            summary[key] = value
        elif isinstance(value, str):
            summary[key] = value
        elif isinstance(value, (list, tuple)):
            preview = list(value[:max_values])
            summary[key] = {"len": len(value), "preview": preview}
        elif isinstance(value, dict):
            summary[key] = {"keys": list(value.keys())}
        else:
            summary[key] = {"type": type(value).__name__}
    return summary




import os, glob, json
import torch

def _find_latest_dump_dir(sample_dir: str, stage: str, branch: str):
    pat = os.path.join(sample_dir, f"*__{stage}__{branch}")
    cands = sorted([d for d in glob.glob(pat) if os.path.isdir(d)])
    return cands[-1] if cands else None

def _load_tensor_dump_root(dump_dir: str):
    pt = os.path.join(dump_dir, "data", "root.pt")
    if not os.path.exists(pt):
        raise FileNotFoundError(f"Missing tensor root.pt: {pt}")
    return torch.load(pt, map_location="cpu", weights_only=False)

def _load_value_from_root_dict(root_dict_dir: str, key: str):
    pt = os.path.join(root_dict_dir, f"{key}.pt")
    js = os.path.join(root_dict_dir, f"{key}.json")

    if os.path.exists(pt):
        return torch.load(pt, map_location="cpu", weights_only=False)
    if os.path.exists(js):
        with open(js, "r", encoding="utf-8") as f:
            return json.load(f)["value"]
    raise KeyError(f"Key '{key}' not found under {root_dict_dir}")

def _load_stage_tensor(dump_root: str, pair_id: int, stage: str, branch: str):
    sample_dir = os.path.join(dump_root, f"sample_{int(pair_id)}")
    if not os.path.isdir(sample_dir):
        raise FileNotFoundError(f"sample_dir not found: {sample_dir}")

    d = _find_latest_dump_dir(sample_dir, stage=stage, branch=branch)
    if d is None:
        raise FileNotFoundError(f"dump not found: sample={pair_id} stage={stage} branch={branch}")
    return _load_tensor_dump_root(d)

def _load_stage_dict(dump_root: str, pair_id: int, stage: str, branch: str):
    sample_dir = os.path.join(dump_root, f"sample_{int(pair_id)}")
    d = _find_latest_dump_dir(sample_dir, stage=stage, branch=branch)
    if d is None:
        raise FileNotFoundError(f"dump not found: sample={pair_id} stage={stage} branch={branch}")
    root_dict = os.path.join(d, "data", "root__dict")
    if not os.path.isdir(root_dict):
        raise FileNotFoundError(f"Missing root__dict: {root_dict}")
    return root_dict

def _load_saved_payload(args, device):
    # ====== 你第一框架 dumper 的根目录（注意要指到 pid_xxxx 那层） ======
    # 例如：dump_dataset/pid_1528321
    dump_root = (
        getattr(args, "diffsynth_dump_root", None)
        or getattr(args, "save_inputs_dir", None)
        or getattr(args, "save_dumps_dir", None)
    )
    if dump_root is None:
        raise ValueError("Please set args.diffsynth_dump_root to your dumper root, e.g. dump_dataset/pid_xxx")

    # ====== 选择你要对齐的 sample ======
    pair_id = int(getattr(args, "saved_pair_id", 0))

    dp_rank = int(mpu.get_data_parallel_rank())
    tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()

    if torch.distributed.get_rank() == tp_cp_src_rank:
        # -------- 1) 从 chosen/rejected_inputs 里拿 context / clip_feature / y --------
        chosen_root = _load_stage_dict(dump_root, pair_id, stage="chosen_inputs", branch="chosen")
        rejected_root = _load_stage_dict(dump_root, pair_id, stage="rejected_inputs", branch="rejected")

        context = _load_value_from_root_dict(chosen_root, "context")

        # 你截图里 y/clip_feature 的命名可能不完全一样，先用候选名兜底
        def _pick(root, candidates):
            last_err = None
            for k in candidates:
                try:
                    return _load_value_from_root_dict(root, k)
                except Exception as e:
                    last_err = e
            raise last_err

        chosen_clip_feature = _pick(chosen_root, ["clip_feature", "chosen_clip_feature", "clip_feat"])
        reject_clip_feature = _pick(rejected_root, ["clip_feature", "reject_clip_feature", "rejected_clip_feature", "clip_feat"])
        chosen_y = _pick(chosen_root, ["y", "chosen_y"])
        reject_y = _pick(rejected_root, ["y", "reject_y", "rejected_y"])

        # -------- 2) 从 training_loss dump 里拿 timestep/noisy_latents/target/weight/noise_pred --------
        
        timestep_chosen = _load_stage_tensor(dump_root, pair_id, "training_loss__timesteps", "chosen")
        timestep_rejected = _load_stage_tensor(dump_root, pair_id, "training_loss__timesteps", "rejected")
        chosen_noisy_latents   = _load_stage_tensor(dump_root, pair_id, stage="training_loss__noisy_latents", branch="chosen")
        reject_noisy_latents   = _load_stage_tensor(dump_root, pair_id, stage="training_loss__noisy_latents", branch="rejected")

        chosen_training_target = _load_stage_tensor(dump_root, pair_id, stage="training_loss__training_target", branch="chosen")
        reject_training_target = _load_stage_tensor(dump_root, pair_id, stage="training_loss__training_target", branch="rejected")

        chosen_loss_weight     = _load_stage_tensor(dump_root, pair_id, stage="training_loss__loss_weight", branch="chosen")
        reject_loss_weight     = _load_stage_tensor(dump_root, pair_id, stage="training_loss__loss_weight", branch="rejected")

        # 可选：对比用（如果你 dump 了）
        try:
            chosen_noise_pred = _load_stage_tensor(dump_root, pair_id, stage="training_loss__noise_pred", branch="chosen")
        except Exception:
            chosen_noise_pred = None
        try:
            reject_noise_pred = _load_stage_tensor(dump_root, pair_id, stage="training_loss__noise_pred", branch="rejected")
        except Exception:
            reject_noise_pred = None

        payload = {
            "context": context,
            "chosen": {
                "clip_feature": chosen_clip_feature,
                "timestep": timestep_chosen,
                "y": chosen_y,
                "noisy_latents": chosen_noisy_latents,
                "training_target": chosen_training_target,
                "loss_weight": chosen_loss_weight,
                "noise_pred": chosen_noise_pred,
            },
            "rejected": {
                "timestep": timestep_rejected,
                "clip_feature": reject_clip_feature,
                "y": reject_y,
                "noisy_latents": reject_noisy_latents,
                "training_target": reject_training_target,
                "loss_weight": reject_loss_weight,
                "noise_pred": reject_noise_pred,
            },
        }

        print(f"[DPO load from dumper] dp_rank={dp_rank} pair_id={pair_id} root={dump_root}")
        print(f"[DPO load] keys={sorted(list(payload.keys()))}")
        print(f"[DPO load] chosen keys={sorted(list(payload['chosen'].keys()))}")
        print(f"[DPO load] rejected keys={sorted(list(payload['rejected'].keys()))}")

    else:
        payload = None

    # 广播到所有 rank（保持你原有逻辑）
    payload = _broadcast_saved_payload(payload, device=device)

    # ====== 按你原函数签名返回 ======
    context = payload["context"]
    chosen_clip_feature = payload["chosen"]["clip_feature"]
    reject_clip_feature = payload["rejected"]["clip_feature"]
    chosen_y = payload["chosen"]["y"]
    reject_y = payload["rejected"]["y"]

    timestep = payload["timestep"].to(dtype=torch.bfloat16, device=device)
    return payload, context, chosen_clip_feature, reject_clip_feature, chosen_y, reject_y, timestep_chosen, timestep_rejected



def _load_batch_inputs(batch, chosen_key, rejected_key):
    context = batch["context"]  # shared text context
    chosen_latents = batch[chosen_key]["latents"]
    reject_latents = batch[rejected_key]["latents"]
    chosen_clip_feature = batch[chosen_key].get(
        "img_clip_feature", batch[chosen_key].get("clip_feature")
    )
    reject_clip_feature = batch[rejected_key].get(
        "img_clip_feature", batch[rejected_key].get("clip_feature")
    )
    chosen_y = batch[chosen_key].get("img_emb_y")
    reject_y = batch[rejected_key].get("img_emb_y")
    return (
        context,
        chosen_latents,
        reject_latents,
        chosen_clip_feature,
        reject_clip_feature,
        chosen_y,
        reject_y,
    )


def _broadcast_tensor(input_tensor: torch.Tensor):
    tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
    if mpu.get_tensor_context_parallel_world_size() > 1:
        dist.broadcast(
            input_tensor,
            tp_cp_src_rank,
            group=mpu.get_tensor_context_parallel_group(),
        )


def _sample_timestep(flow_scheduler, diffusion_config, device):
    min_timestep_boundary = int(
        diffusion_config.get("min_timestep_boundary")
        * flow_scheduler.num_train_timesteps
    )
    max_timestep_boundary = int(
        diffusion_config.get("max_timestep_boundary")
        * flow_scheduler.num_train_timesteps
    )
    timestep_range = [min_timestep_boundary, max_timestep_boundary]
    timestep_id = torch.randint(timestep_range[0], timestep_range[1], (1,))
    timestep = flow_scheduler.timesteps[timestep_id].to(
        dtype=torch.bfloat16,
        device=device,
    )
    return timestep, timestep_id, timestep_range


def _generate_noise(chosen_latents, reject_latents, base_seed, curr_iter):
    if base_seed is not None:
        # 计算当前的随机种子
        seed_chosen = int(base_seed) + curr_iter * 2
        seed_reject = int(base_seed) + curr_iter * 2 + 1
        
        # 打印日志 (如果你在多卡训练，建议加上 rank 区分)
        print(
            f"[Rank {torch.distributed.get_rank()}] noise seeds: "
            f"chosen={seed_chosen}, reject={seed_reject}"
        )
        
        # 创建 Generator
        gen_chosen = torch.Generator(device=chosen_latents.device).manual_seed(seed_chosen)
        gen_reject = torch.Generator(device=reject_latents.device).manual_seed(seed_reject)
        
        # 【修正点】：使用 torch.randn 替代 randn_like
        noise_chosen = torch.randn(
            chosen_latents.shape, 
            device=chosen_latents.device, 
            dtype=chosen_latents.dtype, 
            generator=gen_chosen
        )
        noise_reject = torch.randn(
            reject_latents.shape, 
            device=reject_latents.device, 
            dtype=reject_latents.dtype, 
            generator=gen_reject
        )
    else:
        # 如果没有 seed，直接使用默认的 randn_like 即可
        noise_chosen = torch.randn_like(chosen_latents)
        noise_reject = torch.randn_like(reject_latents)
        
    return noise_chosen, noise_reject


def forward_step(data_iterator, model, time_step=None):
    flow_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True, num_train_timesteps=1000)
    flow_scheduler.set_timesteps(1000, training=True)

    dataset_config = set_config().get("dataset", {})
    chosen_key = dataset_config.get("chosen_video_key", "chosen")
    rejected_key = dataset_config.get("rejected_video_key", "rejected")

    timers = get_timers()
    timers.start_timer('get-data-time')
    batch = next(data_iterator)
    timers.stop_timer('get-data-time')

    use_saved_inputs = bool(getattr(args, "use_saved_inputs", False))
    save_dumps = bool(getattr(args, "save_dumps", False))
    compare_saved_losses = bool(getattr(args, "compare_saved_losses", False))

    if use_saved_inputs:
        # from diffsynth dumps
        (
            payload,
            context,
            chosen_clip_feature,
            reject_clip_feature,
            chosen_y,
            reject_y,
            timestep_c,
            timestep_r,
        ) = _load_saved_payload(args, device=torch.cuda.current_device())
    else:
        payload = None
        (
            context,
            chosen_latents,
            reject_latents,
            chosen_clip_feature,
            reject_clip_feature,
            chosen_y,
            reject_y,
        ) = _load_batch_inputs(batch, chosen_key, rejected_key)
    # =========================
    # timestep sampling
    # =========================
    diffusion_config = (
        set_config()
        .get("model_config", {})
        .get("training", {})
        .get("diffusion", {})
    )

    timestep_range = None
    timestep_id = None
    if not use_saved_inputs:
        timestep_c, timestep_id_c, timestep_range = _sample_timestep(
        flow_scheduler,
        diffusion_config,
        device=torch.cuda.current_device(),
    )
        timestep_r, timestep_id_r, _ = _sample_timestep(
            flow_scheduler,
            diffusion_config,
            device=torch.cuda.current_device(),
        )

    _broadcast_tensor(timestep_c)
    _broadcast_tensor(timestep_r)
    if dist.get_rank() == mpu.get_tensor_context_parallel_src_rank():
        print(f"[Rank {dist.get_rank()}] timestep_c={timestep_c}, id_c={timestep_id_c}")
        print(f"[Rank {dist.get_rank()}] timestep_r={timestep_r}, id_r={timestep_id_r}")
    # =========================
    # noise (chosen/reject independent)
    # =========================
    if not use_saved_inputs:
        base_seed = getattr(args, "noise_seed", None)
        curr_iter = int(getattr(args, "curr_iteration", 0))
        noise_chosen, noise_reject = _generate_noise(
            chosen_latents,
            reject_latents,
            base_seed=base_seed,
            curr_iter=curr_iter,
        )

        _broadcast_tensor(noise_chosen)
        _broadcast_tensor(noise_reject)
    # =========================
    # forward & loss
    # =========================
    return_debug = bool(save_dumps)

    if use_saved_inputs:
        chosen_noisy_latents = payload["chosen"]["noisy_latents"]
        chosen_training_target = payload["chosen"]["training_target"]
        chosen_loss_weight = payload["chosen"]["loss_weight"]
        chosen_noise_pred = payload["chosen"].get("noise_pred")
        reject_noisy_latents = payload["rejected"]["noisy_latents"]
        reject_training_target = payload["rejected"]["training_target"]
        reject_loss_weight = payload["rejected"]["loss_weight"]
        reject_noise_pred = payload["rejected"].get("noise_pred")
        input_checks = {
            "chosen.timestep": (payload["chosen"]["timestep"], timestep_c),
            "rejected.timestep": (payload["rejected"]["timestep"], timestep_r),
            "context": (payload["context"], context),
            "chosen.clip_feature": (payload["chosen"]["clip_feature"], chosen_clip_feature),
            "chosen.y": (payload["chosen"]["y"], chosen_y),
            "chosen.noisy_latents": (payload["chosen"]["noisy_latents"], chosen_noisy_latents),
            "chosen.training_target": (payload["chosen"]["training_target"], chosen_training_target),
            "chosen.loss_weight": (payload["chosen"]["loss_weight"], chosen_loss_weight),
            "rejected.clip_feature": (payload["rejected"]["clip_feature"], reject_clip_feature),
            "rejected.y": (payload["rejected"]["y"], reject_y),
            "rejected.noisy_latents": (payload["rejected"]["noisy_latents"], reject_noisy_latents),
            "rejected.training_target": (payload["rejected"]["training_target"], reject_training_target),
            "rejected.loss_weight": (payload["rejected"]["loss_weight"], reject_loss_weight),
        }
        mismatches = _check_payload_inputs(
            payload,
            input_checks,
            rtol=float(getattr(args, "compare_losses_rtol", 1e-5)),
            atol=float(getattr(args, "compare_losses_atol", 1e-8)),
        )
        if mismatches:
            print(
                f"[DPO input-check] rank={dist.get_rank()} mismatches={mismatches}"
            )
        elif dist.get_rank() == mpu.get_tensor_context_parallel_src_rank():
            print(f"[DPO input-check] rank={dist.get_rank()} ok")

    print(f"[Rank {torch.distributed.get_rank()}] enter chosen compute_single_loss========")
    if use_saved_inputs:
        loss_chosen, loss_wo_w_chosen, first_frame_loss_chosen = (
            _compute_single_loss_from_saved(
                chosen_noisy_latents,
                chosen_training_target,
                chosen_loss_weight,
                context,
                chosen_clip_feature,
                chosen_y,
                model,
                timestep_c,
                expected_noise_pred=chosen_noise_pred,
                compare_name="chosen.noise_pred",
                rtol=float(getattr(args, "compare_losses_rtol", 1e-5)),
                atol=float(getattr(args, "compare_losses_atol", 1e-8)),
            )
        )
    elif return_debug:
        loss_chosen, loss_wo_w_chosen, first_frame_loss_chosen, debug_chosen = (
            _compute_single_loss(
                chosen_latents,
                context,
                chosen_clip_feature,
                chosen_y,
                flow_scheduler,
                model,
                timestep_c,
                noise_chosen,
                return_debug=True,
            )
        )
    else:
        loss_chosen, loss_wo_w_chosen, first_frame_loss_chosen = (
            _compute_single_loss(
                chosen_latents,
                context,
                chosen_clip_feature,
                chosen_y,
                flow_scheduler,
                model,
                timestep_c,
                noise_chosen,
            )
        )

    print(f"[Rank {torch.distributed.get_rank()}] enter reject compute_single_loss========")
    if use_saved_inputs:
        loss_reject, loss_wo_w_reject, first_frame_loss_reject = (
            _compute_single_loss_from_saved(
                reject_noisy_latents,
                reject_training_target,
                reject_loss_weight,
                context,
                reject_clip_feature,
                reject_y,
                model,
                timestep_r,
                expected_noise_pred=reject_noise_pred,
                compare_name="rejected.noise_pred",
                rtol=float(getattr(args, "compare_losses_rtol", 1e-5)),
                atol=float(getattr(args, "compare_losses_atol", 1e-8)),
            )
        )
    elif return_debug:
        loss_reject, loss_wo_w_reject, first_frame_loss_reject, debug_reject = (
            _compute_single_loss(
                reject_latents,
                context,
                reject_clip_feature,
                reject_y,
                flow_scheduler,
                model,
                timestep_r,
                noise_reject,
                return_debug=True,
            )
        )
    else:
        loss_reject, loss_wo_w_reject, first_frame_loss_reject = (
            _compute_single_loss(
                reject_latents,
                context,
                reject_clip_feature,
                reject_y,
                flow_scheduler,
                model,
                timestep_r,
                noise_reject,
            )
        )

    beta = float(
        set_config()
        .get("model_config")
        .get("dit")
        .get("train")
        .get("dpo")
        .get("beta")
    )

    advantage = (loss_reject - loss_chosen).clamp(-20, 20)
    # 只用 advantage 的数值算系数，不让系数本身反传（很关键）
    with torch.no_grad():
        # d/dadv [-logsigmoid(beta*adv)] = beta * sigmoid(-beta*adv)
        print(f"beta = {beta}")
        coeff = beta * torch.sigmoid(-beta * advantage)
        # 对应你 dpo_loss 的 mean()
        coeff = coeff / coeff.numel()

    # 两个“等价”的反传项： +coeff*loss_reject  和  -coeff*loss_chosen
    loss_reject_scaled = (coeff * loss_reject).sum()
    loss_chosen_scaled = (-coeff * loss_chosen).sum()

    # 这个只是用来日志/显示，不参与反传（可选）
    dpo_loss_for_log = (-F.logsigmoid(beta * advantage)).mean().detach()

    if return_debug and not args.test_with_pseudo_data and not use_saved_inputs:
        interval = max(1, int(getattr(args, "save_dumps_interval", 1)))
        curr_iter = int(getattr(args, "curr_iteration", 0))
        if curr_iter % interval == 0:
            save_dir = (
                getattr(args, "save_dumps_dir", None)
                or getattr(args, "save_inputs_dir", None)
                or f"../test_data/saved_inputs_{args.test_resolution}"
            )
            os.makedirs(save_dir, exist_ok=True)
            dp_rank = int(mpu.get_data_parallel_rank())
            payload = {
                "meta": {
                    "iter": curr_iter,
                    "rank": int(torch.distributed.get_rank()),
                    "dp_rank": dp_rank,
                },
                "scheduler_params": _summarize_scheduler(flow_scheduler),
                
                "context": context,
                "chosen": {
                    "clip_feature": chosen_clip_feature,
                    "y": chosen_y,
                    "timestep_c": timestep_c,
                    **debug_chosen,
                },
                "rejected": {
                    "clip_feature": reject_clip_feature,
                    "y": reject_y,
                    "timestep_r": timestep_r,
                    **debug_reject,
                },
                "losses": {
                    "loss_chosen": loss_chosen.detach(),
                    "loss_reject": loss_reject.detach(),
                    "dpo_loss": dpo_loss_for_log.detach(),
                    "loss_wo_w_chosen": loss_wo_w_chosen.detach(),
                    "loss_wo_w_reject": loss_wo_w_reject.detach(),
                    "first_frame_loss_chosen": first_frame_loss_chosen.detach(),
                    "first_frame_loss_reject": first_frame_loss_reject.detach(),
                },
            }
            save_path = os.path.join(
                save_dir, f"dpo_inputs_iter{curr_iter}_rank{dp_rank}.pt"
            )
            torch.save(_detach_to_cpu(payload), save_path)

    if use_saved_inputs and compare_saved_losses:
        saved_losses = payload.get("losses", {}) if isinstance(payload, dict) else {}
        current_losses = {
            "loss_chosen": loss_chosen.detach(),
            "loss_reject": loss_reject.detach(),
            "dpo_loss": dpo_loss_for_log.detach(),
            "loss_wo_w_chosen": loss_wo_w_chosen.detach(),
            "loss_wo_w_reject": loss_wo_w_reject.detach(),
            "first_frame_loss_chosen": first_frame_loss_chosen.detach(),
            "first_frame_loss_reject": first_frame_loss_reject.detach(),
        }
        compare = _compare_losses(
            saved_losses,
            current_losses,
            float(getattr(args, "compare_losses_rtol", 1e-5)),
            float(getattr(args, "compare_losses_atol", 1e-8)),
        )
        tp_cp_src_rank = mpu.get_tensor_context_parallel_src_rank()
        if torch.distributed.get_rank() == tp_cp_src_rank:
            curr_iter = int(getattr(args, "curr_iteration", 0))
            dp_rank = int(mpu.get_data_parallel_rank())
            print(f"[DPO compare] iter={curr_iter} dp_rank={dp_rank} results={compare}")
            compare_dir = (
                getattr(args, "save_dumps_dir", None)
                or getattr(args, "save_inputs_dir", None)
                or f"../test_data/saved_inputs_{args.test_resolution}"
            )
            os.makedirs(compare_dir, exist_ok=True)
            compare_payload = {
                "meta": {
                    "iter": curr_iter,
                    "rank": int(torch.distributed.get_rank()),
                    "dp_rank": dp_rank,
                },
                "compare": compare,
            }
            for name, tensor in current_losses.items():
                if torch.is_tensor(tensor):
                    compare_payload[f"current_mean/{name}"] = float(tensor.float().mean().item())
            for name, tensor in saved_losses.items():
                if torch.is_tensor(tensor):
                    compare_payload[f"saved_mean/{name}"] = float(tensor.float().mean().item())
            compare_path = os.path.join(
                compare_dir, f"dpo_loss_compare_iter{curr_iter}_rank{dp_rank}.pt"
            )
            torch.save(compare_payload, compare_path)

    return [
        loss_reject_scaled,
        loss_chosen_scaled,
        loss_reject.detach(),
        loss_chosen.detach(),
        dpo_loss_for_log.detach(),
    ], dpo_loss_func


if __name__ == "__main__":
    args = parse_args(extra_args=extra_args)
    trainer = Trainer(args)
    trainer.pretrain(forward_step_func=forward_step)
