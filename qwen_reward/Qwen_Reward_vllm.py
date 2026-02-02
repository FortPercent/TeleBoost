import os
import json
from pathlib import Path
import time
from qwen_vl_utils import fetch_video
import torch
import time
import logging
from transformers import Qwen2_5_VLForConditionalGeneration
import re
from typing import List, Dict, Any, Union
# 从metadata读取视频文件作为一个batch，并以batch为单位评估奖励
# 传给模型的messages是video_ids而不是video_path

# input dir path
INPUT_FOLDER = Path('/gemini/space/ljm/Wan2.1-main_72btest_for_more/Reward/metadata')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. 读取json文件组装video_ids[这一块有待完善，应该直接读视频path转化]
def process_json_data(data: dict) -> dict:
    """
    处理单个JSON文件内容的函数。
    
    参数:
        data (dict): 从一个JSON文件中读取并解析后的Python字典。
        
    返回:
        dict: 处理完成后的新字典，将被写入到输出文件中。
        
    --- 请在这里根据你的需求修改 ---
    """
    max_pixels=151200
    fps=data["fps"]
    video_path=data["video_path"]
    vision_info = {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "max_pixels": max_pixels,
                        "fps": fps,
                    }
    video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True)
    # print(f"the type of video_input is:{type(video_input)}")
    return video_input

# --- 3. 主执行逻辑 (通常不需要修改这部分) ---
def generate_video_tensor():
    """
    将input_dir中的json文件中的video转化为tensor类型。
    """
    print(f"开始处理文件夹 '{INPUT_FOLDER}' 中的JSON文件...")

    i=0
    # 遍历输入文件夹中的所有文件
    video_ids=[]
    for input_filepath in INPUT_FOLDER.iterdir():
        if i<30:
            # 检查文件是否为JSON文件
            if input_filepath.suffix == '.json':
                try:
                    # 使用 'with' 语句确保文件能被正确关闭
                    # 使用 utf-8 编码读取文件，防止中文等字符乱码
                    with open(input_filepath, 'r', encoding='utf-8') as f_in:
                        # 读取并解析JSON数据
                        original_data = json.load(f_in)
                    
                    # 调用你的处理函数
                    video_id = process_json_data(original_data)
                    video_ids.append(video_id)
                    print(f"成功处理{i+1}个json文件！")
                    i=i+1

                except json.JSONDecodeError:
                    print(f"  [错误] 文件 {input_filepath.name} 不是有效的JSON格式，已跳过。")
                except Exception as e:
                    print(f"  [错误] 处理文件 {input_filepath.name} 时发生未知错误: {e}")
        else:
            print("处理30个json文件！")
            return video_ids

def generate_batch_prompts_for_train(new_files, max_pixels = 360*420, fps = 1.0):
    messages = []
    prompt = create_simple_prompt()
    i=1
    print("开始生成对话...")
    total_time = 0
    for meta_file in new_files:
        start_time = time.time()
        video_file = meta_file.replace("meta_","video_").replace("metadata","videos").replace(".json",".mp4")
        # 检查视频文件是否存在
        assert os.path.exists(video_file), f"Video file not found: {video_file}"
        
        print(f"Processing: {os.path.basename(video_file)}")
        message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url":f"file://{video_file}"},
                            "max_pixels": max_pixels,
                            "fps": fps,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        messages.append(message)
        processing_time=time.time()-start_time
        total_time+=processing_time
        print(f"成功生成第{i}个message，用时{processing_time}")
        i+=1
    print()
    print(f"成功生成{len(messages)}条messages，用时{total_time}")
    print("="*40)
    return messages
        
        
    
    
def generate_batch_prompts(video_dir, max_pixels = 360*420, fps = 1.0):
    batch_prompts=[]
        
    from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
    processor = Qwen2_5_VLProcessor.from_pretrained(qwen_model_path,trust_remote_code=True)
    print(f"成功加载qwen_processor!")
    print()
        
    logger.info(f"Starting generating batch of prompts ...")
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    p=Path(video_dir)
    if not p.is_dir():
        print(f"Error:directory{video_directory} does not exist.")
        raise FileNotFoundError
        
    messages=[]
    prompt = create_simple_prompt()
    i=1
    print("开始生成对话...")
    total_time=0
    for file_path in p.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS:
            start_time=time.time()
            video_path = str(file_path)
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url":f"file://{video_path}"},
                            "max_pixels": max_pixels,
                            "fps": fps,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            messages.append(message)
            processing_time=time.time()-start_time
            total_time+=processing_time
            print(f"成功生成第{i}个message，用时{processing_time}")
            i+=1
            if i>30:
                print(f"成功生成{len(messages)}条对话，用时{total_time}")
                print("="*40)
                return messages   # 每个message的类型是列表，messages=[message1,message2,...]  
                
    

def create_simple_prompt() -> str:
        """创建结构化视频质量评估提示词（界面风格）"""
        return """请你作为一个专业视频质量评估助手，参考以下评分标准和格式，对给定的视频进行多维度质量评估。请严格按照输出格式，以客观、公正、结构化的方式打分。

                评估维度（每项满分100分）：
                1. 视觉审美（Aesthetics）：
                - 参考项：构图是否合理、光影运用是否自然、色彩搭配是否和谐、整体画面是否具有美感。
                - 高分标准：画面构图精妙、光影自然、色彩生动，具备艺术性。
                - 扣分项：画面凌乱、光照极端或失衡、颜色搭配不当或灰暗。

                2. 局部变形（Distortion）：
                - 参考项：人物或物体是否出现异常形态、肢体是否扭曲、是否有结构性突变或失真、是否突然消失。
                - 高分标准：视频中不存在明显变形，物体结构自然、稳定。
                - 扣分项：出现严重扭曲、肢体不合理、局部区域断裂或消失。

                3. 视觉伪影与不一致（Artifacts/Inconsistency）：
                - 参考项：是否存在突变区域、马赛克、色块、条纹、边缘断裂、纹理模糊等问题。
                - 高分标准：无明显视觉瑕疵，画面一致性强。
                - 扣分项：出现视觉伪影或明显瑕疵，视觉体验受到影响。

                4. 清晰度（Sharpness）：
                - 参考项：细节呈现的清晰度，边缘锐利程度，物体是否具备较高的辨识度。
                - 高分标准：画面细节丰富、边缘清晰锐利。
                - 扣分项：整体模糊、边缘不清晰、细节缺失。

                5. 视觉一致性（Consistency）：
                - 参考项：视频内容在时间上的连贯性，是否存在跳帧、镜头突变或画面不稳定等问题。
                - 高分标准：过渡自然，时间逻辑连贯，画面稳定。
                - 扣分项：镜头跳跃明显、物体突然改变状态、画面抖动。

                评分规则：
                - 每个维度评分在 0 ~ 100 范围内，越好越高分。
                - 合计为五项得分的算术平均，保留整数。
                - 对于某项严重失真或效果极差（如严重模糊、强伪影等），请大胆给出低分（例如低于30分）。
                - 每个视频的打分应充分拉开差距，避免视频之间出现“同分”或“几乎同分”情况。
                - 请确保不同维度之间的评分不互相矛盾，确保评分具有可比性与区分度。

                输出格式（严格遵守）：
                dim1:XX分,dim2:XX分,dim3:XX分,dim4:XX分,dim5:XX分,合计:XX分

                风格要求：
                - 禁止输出解释性文字或分析过程。
                - 禁止使用“我认为”、“可能”、“大致”等模糊词语。
                - 输出必须严格按照上述格式，一次性返回评估结果。

                请严格按照输出格式要求，输出且只输出输出格式的内容。请依照以上标准、逻辑和格式，对视频进行结构化质量评估。
                """

class QwenVideoRewardModel:
    def __init__(self, model_path="/gemini/space/Qwen/Qwen2___5-VL-72B-Instruct", device="cuda", torch_dtype=torch.bfloat16):
        self.device = device
        self.torch_dtype = torch_dtype
        self.model_loaded = False
        
        from collections import deque
        self.score_history = deque(maxlen=1000)
        self.reward_stats = {
            'mean': 50.0,
            'std': 10.0,
            'percentiles': [30, 40, 50, 60, 70, 80, 90]
        }
        
        logger.info(f"Loading Qwen model from {model_path} on {device}...")
        
        print("Creating model configuration...")
        
        # 多GPU模式：使用device_map="auto"让模型自动分配
        if device == "auto":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto",  # 自动分配到多个GPU
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            # 单GPU模式
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map={"": device},
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        
        #self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True) #    Qwen2_5VLProcessor
        from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path,trust_remote_code=True)
        
        # # 确保模型在正确设备上
        # if device != "auto":
        #     self.model = self.model.to(device)
        
        self.model_loaded = True
        logger.info(f"Qwen Video Reward Model loaded on {self.model.device}")
        
    def create_simple_prompt(self) -> str:
        """创建结构化视频质量评估提示词（界面风格）"""
        return """请你作为一个专业视频质量评估助手，参考以下评分标准和格式，对给定的视频进行多维度质量评估。请严格按照输出格式，以客观、公正、结构化的方式打分。

                评估维度（每项满分100分）：
                1. 视觉审美（Aesthetics）：
                - 参考项：构图是否合理、光影运用是否自然、色彩搭配是否和谐、整体画面是否具有美感。
                - 高分标准：画面构图精妙、光影自然、色彩生动，具备艺术性。
                - 扣分项：画面凌乱、光照极端或失衡、颜色搭配不当或灰暗。

                2. 局部变形（Distortion）：
                - 参考项：人物或物体是否出现异常形态、肢体是否扭曲、是否有结构性突变或失真、是否突然消失。
                - 高分标准：视频中不存在明显变形，物体结构自然、稳定。
                - 扣分项：出现严重扭曲、肢体不合理、局部区域断裂或消失。

                3. 视觉伪影与不一致（Artifacts/Inconsistency）：
                - 参考项：是否存在突变区域、马赛克、色块、条纹、边缘断裂、纹理模糊等问题。
                - 高分标准：无明显视觉瑕疵，画面一致性强。
                - 扣分项：出现视觉伪影或明显瑕疵，视觉体验受到影响。

                4. 清晰度（Sharpness）：
                - 参考项：细节呈现的清晰度，边缘锐利程度，物体是否具备较高的辨识度。
                - 高分标准：画面细节丰富、边缘清晰锐利。
                - 扣分项：整体模糊、边缘不清晰、细节缺失。

                5. 视觉一致性（Consistency）：
                - 参考项：视频内容在时间上的连贯性，是否存在跳帧、镜头突变或画面不稳定等问题。
                - 高分标准：过渡自然，时间逻辑连贯，画面稳定。
                - 扣分项：镜头跳跃明显、物体突然改变状态、画面抖动。

                评分规则：
                - 每个维度评分在 0 ~ 100 范围内，越好越高分。
                - 合计为五项得分的算术平均，保留整数。
                - 对于某项严重失真或效果极差（如严重模糊、强伪影等），请大胆给出低分（例如低于30分）。
                - 每个视频的打分应充分拉开差距，避免视频之间出现“同分”或“几乎同分”情况。
                - 请确保不同维度之间的评分不互相矛盾，确保评分具有可比性与区分度。

                输出格式（严格遵守）：
                dim1:XX分,dim2:XX分,dim3:XX分,dim4:XX分,dim5:XX分,合计:XX分

                风格要求：
                - 禁止输出解释性文字或分析过程。
                - 禁止使用“我认为”、“可能”、“大致”等模糊词语。
                - 输出必须严格按照上述格式，一次性返回评估结果。

                请严格按照输出格式要求，输出且只输出输出格式的内容。请依照以上标准、逻辑和格式，对视频进行结构化质量评估。
                """
    
      
    def generate_batch_prompts(self, video_ids, max_pixels = 360*420, fps = 1.0):
        batch_prompts=[]
        
        logger.info(f"Starting generating batch of prompts ...")
        
        prompt = self.create_simple_prompt()
        i=1
        start_time=time.time()
        for video_id in video_ids:
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_id,
                                "max_pixels": max_pixels,
                                "fps": fps,
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            text = self.processor.apply_chat_template(
                messages, tokenize = False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images = None,
                videos = [video_id],
                padding = True,
                return_tensors = 'pt',
            )
            inputs = inputs.to(self.model.device)
            batch_prompts.append(inputs)
            print(f"successfully generating {i+1} prompts, consuming time {time.time()-start_time}s")
            i=i+1
            start_time=time.time()
        return batch_prompts  #[prompt_1,prompt_2,...]
       
    def evaluate_video_batch_reward(self, batch_prompts):
        logger.info("generating evaluation results...")
        i=0
        total_time=0
        results=[]
        for prompts in batch_prompts:
            start_time=time.time()
            i = i+1
            # 推理：生成输出
            with torch.no_grad():
                generated_ids = self.model.generate(**prompts, max_new_tokens=64,do_sample=False, 
                                                    #temperature=0.1
                                                    )
            generated_ids_trimmed =  [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(prompts.input_ids, generated_ids)
            ]
        
            output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]   
            
            logger.info(f"Generated text length of the {i}-th prompt:{len(output_text)}")
            logger.info(f"Generated text preview of the {i}-th prompt:{output_text[:200]}...")
            
            # 解析结果
            result = self.parse_simple_evaluation(output_text)
        
            # 添加处理时间
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            logger.info(f"Generating the {i}-th reward consumes {processing_time}s")
            print("="*30)
            
            results.append(result)
            total_time += processing_time
            
        logger.info(f"⏱️ Video evaluation completed in {total_time:.2f} seconds")
        return results
    
    def parse_simple_evaluation(self, output_text: str) -> Dict[str, Any]:
        """解析简单评估结果"""
        # 针对新格式的分数提取模式
        score_patterns = [
            r'合计[：:]\s*(\d+(?:\.\d+)?)\s*分',          # 合计：80分
            r'综合得分[：:]\s*(\d+(?:\.\d+)?)\s*分',      # 综合得分：80分
            r'总分[：:]\s*(\d+(?:\.\d+)?)\s*分',          # 总分：80分
            r'最终[：:]\s*(\d+(?:\.\d+)?)\s*分',          # 最终：80分
            r'评分[：:]\s*(\d+(?:\.\d+)?)\s*分',          # 评分：80分
            r'(\d+(?:\.\d+)?)\s*分',                     # 80分
            r'(\d+(?:\.\d+)?)/100',                      # 80/100
        ]
        
        score = 50.0  # 默认分数
        for pattern in score_patterns:
            match = re.search(pattern, output_text)
            if match:
                found_score = float(match.group(1))
                if found_score > 100:
                    found_score = min(found_score, 100)
                score = found_score
                logger.info(f"Found score: {score}")
                break
        
        return {
            "overall_score": score,
            "raw_output": output_text
        }
    
    def compute_adaptive_reward(self, raw_score: float) -> float:
        """基于历史分布的自适应奖励计算"""
        # 更新历史统计
        self.score_history.append(raw_score)
        return float(raw_score)  

def parse_simple_evaluation(output_text: str) -> Dict[str, Any]:
    """解析简单评估结果 - 匹配新的prompt格式"""
    # 针对新格式的分数提取模式
    score_patterns = [
            r'合计[：:]\s*(\d+(?:\.\d+)?)\s*分',          # 合计：80分
            r'综合得分[：:]\s*(\d+(?:\.\d+)?)\s*分',      # 综合得分：80分
            r'总分[：:]\s*(\d+(?:\.\d+)?)\s*分',          # 总分：80分
            r'dim5[：:]\s*(\d+(?:\.\d+)?)\s*分.*?合计[：:]\s*(\d+(?:\.\d+)?)\s*分',  # 提取合计分数
            r'最终[：:]\s*(\d+(?:\.\d+)?)\s*分',          # 最终：80分
            r'评分[：:]\s*(\d+(?:\.\d+)?)\s*分',          # 评分：80分
            r'质量评分[：:]\s*(\d+(?:\.\d+)?)',          # 质量评分：80
            r'分数[：:]\s*(\d+(?:\.\d+)?)',              # 分数：80
            r'(\d+(?:\.\d+)?)\s*分',                     # 80分
            r'(\d+(?:\.\d+)?)/100',                      # 80/100
            r'(\d+(?:\.\d+)?)%',                         # 80%
        ]
        
    score = 50.0  # 默认分数
    for pattern in score_patterns:
        match = re.search(pattern, output_text)
        if match:
            # 对于有多个捕获组的模式，取最后一个（合计分数）
            if len(match.groups()) > 1:
                found_score = float(match.group(2))  # 取合计分数
            else:
                found_score = float(match.group(1))
                
            if found_score > 100:
                found_score = min(found_score, 100)
            score = found_score
            logger.info(f"Found score: {score} using pattern: {pattern}")
            break
        
    # 如果没找到分数，记录日志
    if score == 50.0:
        logger.warning(f"No score found in output, using default 50.0. Output: {output_text[:200]}...")
        
    # 尝试提取各个维度的分数
    dimension_scores = {}
    dim_patterns = [
            (r'dim1[：:]\s*(\d+(?:\.\d+)?)\s*分', 'visual_artifacts'),
            (r'dim2[：:]\s*(\d+(?:\.\d+)?)\s*分', 'local_deformation'),
            (r'dim3[：:]\s*(\d+(?:\.\d+)?)\s*分', 'noise_quality'),
            (r'dim4[：:]\s*(\d+(?:\.\d+)?)\s*分', 'clarity_sharpness'),
            (r'dim5[：:]\s*(\d+(?:\.\d+)?)\s*分', 'color_accuracy'),
        ]
        
    for pattern, dim_name in dim_patterns:
        match = re.search(pattern, output_text)
        if match:
            dimension_scores[dim_name] = float(match.group(1))
        
    result = {
            "overall_score": score,
            "summary": output_text[:500] + "..." if len(output_text) > 500 else output_text,
            "raw_output": output_text
        }
        
    # 如果提取到了维度分数，也加入结果
    if dimension_scores:
        result["dimension_scores"] = dimension_scores
        logger.info(f"Extracted dimension scores: {dimension_scores}")
        
    return result             

def get_batch_results(outputs): 
    """
    from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
    processor = Qwen2_5_VLProcessor.from_pretrained(qwen_model_path,trust_remote_code=True)
    print(f"成功加载qwen_processor!")
    """
    results = []
    for single_prompt_output in outputs:
        for response in single_prompt_output:
            output = response.outputs[0]
            output_text = output.text
            logger.info(f"Generated text length: {len(output_text)}")
            logger.info(f"Generated text preview: {output_text[:200]}...")
        
            # 解析结果
            result = parse_simple_evaluation(output_text)
            overall_score = result.get("overall_score","N/A")
            # processing_time = result.get("processing_time", 0)
    
            print(f"🎯 综合质量分数: {overall_score}/100")
            # print(f"⏱️ 处理时间: {processing_time:.2f} 秒")
            print()
            results.append(result)
    return results            

# 当该脚本被直接运行时，才执行main函数
if __name__ == "__main__":
    # 生成video_ids
    # video_ids=generate_video_tensor()
    # print("------video_id类型是：--------")
    # print(type(video_ids[0]))
    # print("video_ids长度是：")
    # print(len(video_ids))
    
    # 加载奖励模型
    qwen_model_path = "/gemini/space/Qwen/Qwen2___5-VL-72B-Instruct"
    video_dir = "/gemini/space/ljm/Wan2.1-main_72btest_for_more/Reward/videos"
    # 生成传给奖励模型的prompts列表
    messages = generate_batch_prompts(video_dir)
    print(f"得到的message类型是{type(messages[0])}")
    print(f"成功加载{len(messages)}个prompts")
    
    print(f"从{qwen_model_path}加载processor...")
    
    print("使用processor预处理数据")
    from vllm import LLM, SamplingParams
    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.90)
    print(f"Loading Qwen model from {qwen_model_path}")
    
    llm = LLM(model = qwen_model_path, tensor_parallel_size = 4, gpu_memory_utilization = 0.90, allowed_local_media_path = video_dir,)
    
    i=0
    print("="*40)
    print("开始用vllm计算得分")
    outputs=[]
    """
    # batch_prompts是messages组成的列表
    for batch_prompt in batch_prompts:
        output = llm.generate(batch_prompt.input_ids, sampling_params)
        
        outputs.append(output)
    """
    start_time=time.time()
    i=1
    print(len(messages))
    
    for message in messages:
        output = llm.chat(message, sampling_params= sampling_params)
        print(f"成功处理第{i}个message")
        i+=1
        outputs.append(output)
    
    print("得到得分列表")
    print(f"得到的outputs类型：{type(outputs)}")
    print(f"得到的outputs长度：{len(outputs)}")
    single_prompt_output = outputs[0]  
    print(f"得到的single_prompt_output类型：{type(single_prompt_output)}")
    print(f"single_prompt_output的长度为{len(single_prompt_output)}")
    response = single_prompt_output[0]
    print(f"得到的response类型：{type(response)}")
    output = response.outputs[0]
    print(f"得到的output类型：{type(output)}")
    text = output.text
    print(f"得到的text类型：{type(text)}")
    # print(f"得到的output.outputs.text类型")
    # exit(0)
    results = get_batch_results(outputs)
    total_time=time.time()-start_time
    print(f"用时：{total_time}")
    
   
    
    """
    for output in outputs:
        i+=1
        print(f"Generate {i}-th output from Wan model")
        print(len(output))
        # print(output[-100:])
    """