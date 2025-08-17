# -*- coding: utf-8 -*-
"""
Kronos时间序列预测模型

该模块实现了基于Kronos架构的时间序列预测模型，专为金融时间序列预测设计，
并与Qlib框架集成。

Author: Hugo
Date: 2025-08-13 12:14:10
LastEditors: shen.lan123@gmail.com
LastEditTime: 2025-08-15 14:42:22
"""

import torch
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np

from ....model.base import Model
from ....log import get_module_logger
from ..pytorch_utils import count_parameters
from ....data.dataset.handler import DataHandlerLP
from .kronos_model import Kronos, KronosTokenizer
from .kronos_model.kronos import auto_regressive_inference,calc_time_stamps

class DailyBatchSampler(Sampler):
    """按日期分批的采样器
    
    该采样器将具有相同日期的样本分组到同一批次中，确保在同一批次中处理同一天的所有股票数据。
    """
    
    def __init__(self, data_source):
        """初始化DailyBatchSampler
        
        Args:
            data_source: 数据源，应包含get_index()方法返回索引信息
        """
        self.data_source = data_source
        # 计算每天的样本数量
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        # 计算每批的起始索引
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    def __iter__(self):
        """迭代器，为每个批次生成索引"""
        for idx, count in zip(self.daily_index, self.daily_count):
            yield np.arange(idx, idx + count)

    def __len__(self):
        """返回数据源的长度"""
        return len(self.data_source)

def build_index_windows(dataset, index: pd.DatetimeIndex, step_len: int, pred_len: int, freq: str = "D", col_set: list = ["feature"])->Dict[Tuple, Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """构建从样本索引（日期，股票代码）到（历史索引，未来索引）的映射
    
    该函数为每个样本构建历史时间窗口和未来时间窗口的索引，用于时间序列预测。
    
    Parameters
    ----------
    dataset : DatasetH / TSDatasetH
        数据集对象，用于获取完整的时间序列数据（通过 dataset.handler.fetch）。
    index : pd.DatetimeIndex
        测试数据集的索引序列，通常包含 (instrument, datetime) 元组。
    step_len : int
        历史窗口长度，即向后回溯的数据点数量。
    pred_len : int
        预测步数，即向前预测的时间点数量。
    freq : str, optional
        时间频率，用于在末端数据不足时生成未来时间索引，默认为 "D"。
    col_set : list, optional
        传递给 handler.fetch 的列集合，默认为 ["feature"]。
        
    Returns
    -------
    Dict[Tuple, Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]
        映射字典，键为 (日期, 股票代码) 元组，值为 (历史时间索引, 未来时间索引) 元组。
        
    Raises
    ------
    TypeError
        当index不是pd.MultiIndex类型时抛出。
    KeyError
        当指定日期在完整时间索引中找不到时抛出。
    """
    # 获取完整时间范围
    start = dataset.get_min_time(dataset.segments)
    end = dataset.get_max_time(dataset.segments)

    # 获取完整的时间序列数据
    full_df: pd.DataFrame = dataset.handler.fetch(slice(start, end), col_set=col_set)
    # 获取去重后的时间索引
    full_idx: pd.DatetimeIndex = full_df.index.get_level_values("datetime").drop_duplicates()
    total_size: int = len(full_idx) - 1

    idx_dict = {}
    
    if not isinstance(index, pd.MultiIndex):
        raise TypeError("index must be a pd.MultiIndex")
    
    for item in index:
        date = item[0]
        inst = item[1]
        key = (date, inst)

        try:
            target_idx: int = full_idx.get_loc(date)
        except KeyError:
            # 日期不在完整时间索引中
            raise KeyError(f"Date {date} not found in full_idx.")

        # 计算历史时间窗口（尽量保证不越界）
        start_idx = max(0, target_idx - step_len + 1)
        hist_idx = full_idx[start_idx : target_idx + 1]

        # 计算未来时间窗口
        if target_idx >= total_size:
            # 如果已到末尾，则用bdate_range/date_range生成
        
            forward_idx = pd.bdate_range(start=date, periods=pred_len+1, freq=freq)[1:]
           
        else:
            # 否则从完整时间索引中切片
            forward_idx = full_idx[target_idx + 1 : target_idx + pred_len + 1]

        idx_dict[key] = (hist_idx, forward_idx)

    return idx_dict

class KronosTS(Model):
    """Kronos时间序列预测模型类
    
    该类实现了基于Kronos架构的时间序列预测模型，专为金融时间序列预测设计，
    并与Qlib框架集成。模型使用预训练的Kronos权重，支持批量处理多个股票。
    """
    
    def __init__(
        self,
        max_context: int = 512,
        clip: int = 5,
        model_name: str = "NeoQuasar/Kronos-base",
        tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        force_download: bool = False,
        local_files_only: bool = False,
        trust_remote_code: bool = True,
        pred_len: int = 1,
        freq: str = "D",
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 1,
        GPU: int = 0,
        output_mode: str = "return",  # 'return' | 'price'
    ) -> None:
        """初始化KronosTS模型
        
        Parameters
        ----------
        max_context : int, optional
            模型的最大上下文长度，默认为512
        clip : int, optional
            输入数据的裁剪值，默认为5
        model_name : str, optional
            Hugging Face模型名称，默认为"NeoQuasar/Kronos-base"
        tokenizer_name : str, optional
            Hugging Face分词器名称，默认为"NeoQuasar/Kronos-Tokenizer-base"
        force_download : bool, optional
            是否强制下载模型，默认为False
        local_files_only : bool, optional
            是否仅使用本地文件，默认为False
        trust_remote_code : bool, optional
            是否信任远程代码，默认为True
        pred_len : int, optional
            预测长度，默认为1
        freq : str, optional
            时间频率，默认为"D"
        temperature : float, optional
            采样温度，默认为1.0
        top_k : int, optional
            Top-K采样参数，默认为0
        top_p : float, optional
            Top-P采样参数，默认为0.9
        sample_count : int, optional
            采样次数，默认为1
        GPU : int, optional
            GPU设备编号，默认为0
        output_mode : str, optional
            输出模式，可选'return'或'price'，默认为'return'
        """
        self.logger = get_module_logger("KronosTS")
        self.logger.info("Initializing KronosTS (pretrained Kronos wrapper)...")

        self.max_context = max_context
        self.clip = clip
        self.pred_len = pred_len
        self.freq = freq
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.sample_count = sample_count
        self.output_mode = output_mode.lower()
        if self.output_mode not in {"return", "price"}:
            raise ValueError("output_mode must be 'return' or 'price'")
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() and GPU >= 0 else "cpu")

        try:
            self.tokenizer = KronosTokenizer.from_pretrained(
                tokenizer_name,
                force_download=force_download,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:  # noqa
            self.logger.warning(
                f"Failed to load tokenizer '{tokenizer_name}' from hub: {e}. Fallback random init."
            )
            self.tokenizer = KronosTokenizer(
                d_in=6,
                d_model=256,
                n_heads=8,
                ff_dim=512,
                n_enc_layers=2,
                n_dec_layers=2,
                ffn_dropout_p=0.0,
                attn_dropout_p=0.0,
                resid_dropout_p=0.0,
                s1_bits=8,
                s2_bits=8,
                beta=0.0,
                gamma0=0.0,
                gamma=0.0,
                zeta=0.0,
                group_size=1,
            )

        try:
            self.Kronos_model = Kronos.from_pretrained(
                model_name,
                force_download=force_download,
                local_files_only=local_files_only
            )
        except Exception as e:  # noqa
            self.logger.warning(
                f"Failed to load model '{model_name}' from hub: {e}. Fallback random init."
            )
            self.Kronos_model = Kronos(
                s1_bits=8,
                s2_bits=8,
                n_layers=2,
                d_model=256,
                n_heads=8,
                ff_dim=512,
                ffn_dropout_p=0.0,
                attn_dropout_p=0.0,
                resid_dropout_p=0.0,
                token_dropout_p=0.0,
                learn_te=False,
            )

        self.logger.info(
            (
                "KronosTS parameters setting:"\
                "\nmax_context : {max_context}"\
                "\nclip : {clip}"\
                "\nmodel_name : {model_name}"\
                "\ntokenizer_name : {tokenizer_name}"\
                "\nGPU : {GPU}"\
                "\ndevice : {device}"\
                "\nuse_gpu : {use_gpu}"\
                "\npred_len : {pred_len}"\
                "\nfrequency : {freq}"\
                "\nT/top_k/top_p/sample_count : {temperature}/{top_k}/{top_p}/{sample_count}"\
                "\noutput_mode : {output_mode}"\
            ).format(
                max_context=max_context,
                clip=clip,
                model_name=model_name,
                tokenizer_name=tokenizer_name,
                GPU=GPU,
                device=self.device,
                use_gpu=self.use_gpu,
                pred_len=self.pred_len,
                freq=self.freq,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                sample_count=self.sample_count,
                output_mode=self.output_mode,
            )
        )
        self.logger.info("model:\n{:}".format(self.Kronos_model))
        self.logger.info(
            "model size: {:.4f} MB".format(count_parameters(self.Kronos_model))
        )
        self.fitted = True
        self._predictor = None
        self.tokenizer.to(self.device)
        self.Kronos_model.to(self.device)
        
    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")
    
    
    def fit(self, *args, **kwargs):  # noqa: D401
        """预训练Kronos模型的空操作拟合方法
        
        Kronos基础模型是生成式/无监督的；此处未实现基于qlib标签的监督微调。
        此方法的存在是为了满足qlib工作流接口的要求。
        
        Parameters
        ----------
        *args : tuple
            位置参数（未使用）
        **kwargs : dict
            关键字参数（未使用）
        """
        self.logger.info("KronosTS uses a pretrained model; skipping training (fit no-op).")
        self.fitted = True

    def _ensure_predictor(self, x, x_stamp, y_stamp):
        """确保预测器的输入数据格式正确并执行预测
        
        Parameters
        ----------
        x : torch.Tensor
            输入的时间序列数据，形状为(batch_size, seq_len, features)
        x_stamp : torch.Tensor
            输入的时间戳数据，形状为(batch_size, x_seq_len, time_features)
        y_stamp : torch.Tensor
            输出的时间戳数据，形状为(batch_size, y_seq_len, time_features)
            
        Returns
        -------
        torch.Tensor
            预测结果，形状为(batch_size, pred_len, features)
        """
        preds = auto_regressive_inference(self.tokenizer, self.Kronos_model, x, x_stamp, y_stamp, self.max_context, self.pred_len,
                                          self.clip, self.temperature, self.top_k, self.top_p, self.sample_count, False)
        preds = preds[:, -self.pred_len:, :]
        return preds
    
    def predict(self, dataset):  # noqa: D401
        """Generate feature for qlib pipeline using pretrained Kronos.

        Workflow:
        1. Extract test samples from dataset (feature+label tensor per index).
        2. For each batch build OHLCV(+amount) DataFrame according to feature_map.
        3. Call KronosPredictor to generate future horizon (pred_len).
        4. Derive feature value per sample:
           - output_mode == 'price': first predicted close price.
           - output_mode == 'return': (first_pred_close / last_hist_close) - 1.

        Returns
        -------
        pd.Series
            Indexed by dataset test index (typically (instrument, datetime)). Values are
            the generated feature (float) or NaN if any step failed.
        Notes
        -----
        - Current implementation supports batch processing for multiple stocks.
        - Assumes evenly spaced timestamps with frequency `self.freq`.
        - Does not use provided label (unsupervised generative inference).
        """
        if not self.fitted:
            raise ValueError("Model not fitted (pretrained load failed).")
        
        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_R)
        # 填充缺失值：先前向填充，再后向填充（config 为就地修改，勿赋值）
        dl_test.config(fillna_type="ffill+bfill")

        # 空数据防护
        try:
            is_empty = getattr(dl_test, "empty", False) or (len(dl_test) == 0)
        except Exception:  # 某些实现可能不支持 len
            is_empty = getattr(dl_test, "empty", False)
        if is_empty:
            self.logger.warning("dl_test is empty; returning empty Series.")
            return pd.Series(dtype=float)

        index = dl_test.get_index()  # MultiIndex (instrument, datetime)
        target_idx:Dict = build_index_windows(dataset, index, dataset.step_len, self.pred_len, self.freq, col_set=["feature"])
        
        preds_list = []  # feature values per sample

        # sampler_test = DailyBatchSampler(dl_test)
        # test_loader = DataLoader(dl_test, sampler=sampler_test, num_workers=1)

        for i in range(len(dl_test)):
            data = dl_test[i]
            data = data[:,:-1]
           
            x_timestamp,y_timestamp = target_idx[index[i]]
            x_time_df = calc_time_stamps(x_timestamp.to_series())
            y_time_df = calc_time_stamps(y_timestamp.to_series())
            x_stamp = x_time_df.values.astype(np.float32)
            y_stamp = y_time_df.values.astype(np.float32)
            
            x = torch.from_numpy(data).to(self.device)
        
            x_stamp = torch.from_numpy(x_time_df.values.astype(np.float32)).to(self.device)
            y_stamp = torch.from_numpy(y_time_df.values.astype(np.float32)).to(self.device)
            x_mean = x.mean(dim=0)
            x_std = x.std(dim=0, unbiased=False)
    

            x = (x - x_mean) / (x_std + 1e-5)
            x = torch.clamp(x, -float(self.clip), float(self.clip))

            x = x.unsqueeze(0)
            x_stamp = x_stamp.unsqueeze(0)
            y_stamp = y_stamp.unsqueeze(0)

            preds = self._ensure_predictor(x, x_stamp, y_stamp)

            preds = preds.squeeze(0)
            preds = preds * (x_std + 1e-5) + x_mean
            
            if self.output_mode=="price":
                preds_list.append(preds[-1, 3])  # 预测的收盘价
            elif self.output_mode=="return":
                
                last_pred_close = preds[-1, 3]  # 预测的收盘价
                last_price = torch.tensor(data[-1, 3], device=last_pred_close.device, dtype=last_pred_close.dtype)
                # print(f"index:{index[i]},pred_close:{last_pred_close},last_price:{last_price}")
                preds_list.append((last_pred_close / last_price) - 1)  # 计算收益率
                
        # 循环结束后、返回前统一转换
        if len(preds_list) == 0:
            return pd.Series(dtype=float)

        # 保证都是 tensor（若存在非 tensor 的异常值，先转换）
        for i, v in enumerate(preds_list):
            if not isinstance(v, torch.Tensor):
                preds_list[i] = torch.tensor(v, device=self.device, dtype=torch.float32)

        # 在设备上 stack（一次性），再 detach->cpu->numpy（一次性传输）
        stacked = torch.stack(preds_list)  # shape (N,)
        arr = stacked.detach().cpu().numpy()
        return pd.Series(arr, index=index)