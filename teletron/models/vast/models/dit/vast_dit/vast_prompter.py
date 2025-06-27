from .base_prompter import BasePrompter
from .vast_video_text_encoder import VastTextEncoder
from transformers import AutoTokenizer
import os, torch
import ftfy
import html
import string
import regex as re


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def canonicalize(text, keep_punctuation_exact_string=None):
    text = text.replace('_', ' ')
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans('', '', string.punctuation))
            for part in text.split(keep_punctuation_exact_string))
    else:
        text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class HuggingfaceTokenizer:
    def __init__(self, name: str, seq_len: int = None, clean: str = None, **kwargs):
        """
        初始化 HuggingfaceTokenizer 实例

        参数:
            name (str): 预训练模型的名称或路径
            seq_len (int, optional): 序列最大长度
            clean (str, optional): 清洗方式，可选值为 None, 'whitespace', 'lower', 'canonicalize'
            **kwargs: 传递给 AutoTokenizer.from_pretrained 的其他参数
        """
        self._validate_clean_option(clean)
        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        # 初始化 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def _validate_clean_option(self, clean: str):
        """验证清洗选项是否合法"""
        if clean not in (None, 'whitespace', 'lower', 'canonicalize'):
            raise ValueError("clean 参数必须为 None, 'whitespace', 'lower' 或 'canonicalize'")

    def _clean_text(self, text: str) -> str:
        """
        根据 clean 选项对文本进行清洗

        参数:
            text (str): 原始文本

        返回:
            str: 清洗后的文本
        """
        if self.clean == 'whitespace':
            return whitespace_clean(basic_clean(text))
        elif self.clean == 'lower':
            return whitespace_clean(basic_clean(text)).lower()
        elif self.clean == 'canonicalize':
            return canonicalize(basic_clean(text))
        return text

    def _tokenize(self, sequence, return_mask, **kwargs):
        """
        对输入序列进行 tokenization

        参数:
            sequence (str or list of str): 输入文本或文本列表
            return_mask (bool): 是否返回 attention mask
            **kwargs: 传递给 tokenizer 的其他参数

        返回:
            Tuple[torch.Tensor, torch.Tensor]: token_ids 和 attention_mask（如果 return_mask 为 True）
        """
        # 设置默认参数
        tokenization_kwargs = {
            'return_tensors': 'pt',
            'padding': 'max_length' if self.seq_len is not None else 'do_not_pad',
            'truncation': True if self.seq_len is not None else False,
            'max_length': self.seq_len
        }
        tokenization_kwargs.update(kwargs)

        # 处理输入为列表
        if isinstance(sequence, str):
            sequence = [sequence]

        # 清洗文本
        if self.clean:
            sequence = [self._clean_text(u) for u in sequence]

        # 执行 tokenization
        tokenized = self.tokenizer(sequence, **tokenization_kwargs)

        # 返回结果
        if return_mask:
            return tokenized.input_ids, tokenized.attention_mask
        else:
            return tokenized.input_ids

    def __call__(self, sequence, **kwargs):
        """
        支持直接调用 tokenizer 实例

        参数:
            sequence (str or list of str): 输入文本或文本列表
            **kwargs: 传递给 _tokenize 方法的参数

        返回:
            Tuple[torch.Tensor, torch.Tensor] 或 torch.Tensor: token_ids 和 attention_mask（如果 return_mask 为 True）
        """
        return self._tokenize(sequence, **kwargs)


class VastPrompter(BasePrompter):
    def __init__(self, tokenizer_path: str = None, text_len: int = 512):
        super().__init__()
        self.text_len = text_len
        self.text_encoder = None
        self.tokenizer = None
        self._initialize_tokenizer(tokenizer_path)

    def _initialize_tokenizer(self, tokenizer_path: str = None):
        """初始化 tokenizer"""
        if tokenizer_path is not None:
            self.tokenizer = HuggingfaceTokenizer(
                name=tokenizer_path,
                seq_len=self.text_len,
                clean='whitespace'
            )

    def fetch_models(self, text_encoder: VastTextEncoder = None):
        """设置文本编码器"""
        self.text_encoder = text_encoder

    def move_to_device(self, obj, device: str = "cuda"):
        """将模型或张量移动到指定设备"""
        return obj.to(device)

    def process_encoded_prompt(self, ids, mask, device: str = "cuda"):
        """处理编码后的 prompt，包括设备移动和截断处理"""
        ids = self.move_to_device(ids, device)
        mask = self.move_to_device(mask, device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def encode_prompt(self, prompt: str, positive: bool = True, device: str = "cuda"):
        """对 prompt 进行编码和处理"""
        prompt = self.process_prompt(prompt, positive=positive)
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        return self.process_encoded_prompt(ids, mask, device)
