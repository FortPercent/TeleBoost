# Weight Conversion

The checkpoint format used by TeleTron is the same as that used by Megatron-LM.

When you want to train models from HuggingFace using TeleTron, you need to convert the HuggingFace checkpoint to the TeleTron format. Conversely, when you want to perform inference using the HuggingFace format after training, you need to convert the TeleTron checkpoint back to the HuggingFace format.

The code for implementing these conversion functions is located in `convert_hunyuanvideo.py`. The scripts `convert_hf2tel.sh` and `convert_tel2hf.sh` are used for converting weights from HuggingFace to TeleTron and from TeleTron to HuggingFace, respectively.
