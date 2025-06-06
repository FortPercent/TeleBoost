from vast.models import  HunyuanVideoTransformer3DModel

path = "/nvfile-heatstorage/hyc/vast/work_dirs/hunyuanvideo_i2vhy_sp2_720p_85_24fps_0512/models/checkpoint_epoch_1_step_5000"
save_path = "/nvfile-heatstorage/teleai-infra/litian/megatron_ckpt/ckpt_tp1_multi_ref_images"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    path,
    allow_pickle=True,
    trust_remote_code=True,    
)
transformer.save_pretrained(save_path)
