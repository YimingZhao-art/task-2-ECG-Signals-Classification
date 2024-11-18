import numpy as np
import pandas as pd
import neurokit2 as nk
from multiprocessing import Pool
from tqdm import tqdm


def fix_length_and_clean(signal):
    # 反转ECG信号并清理
    signal, is_inverted = nk.ecg_invert(signal, sampling_rate=300, show=False)
    c_signal = nk.ecg_clean(signal, sampling_rate=300, method="neurokit")
    # 将信号重采样到固定长度
    fixed_size = nk.signal_resample(
        c_signal,
        desired_length=6000,
        sampling_rate=300,
        desired_sampling_rate=300,
        method="FFT",
    )
    return fixed_size


def get_features(df_raw_signals):
    # 初始化特征列表
    features = []

    # 遍历每个信号，处理并提取特征
    for i in tqdm(range(0, df_raw_signals.shape[0]), desc="Processing signals"):
        signal = df_raw_signals.iloc[i].dropna().to_numpy(dtype="float32")
        f = fix_length_and_clean(signal)
        features.append(f)

    # 将特征转换为DataFrame
    df = pd.DataFrame(features)
    return df


def sub_features(arg_tuple):
    # 提取子数据集并处理特征
    df_raw, idx = arg_tuple
    df_processed = get_features(df_raw)
    return idx, df_processed


def multi_features(df_raw_signals, n_cores=8):
    # 获取信号的索引并分割为多个子集
    ids = df_raw_signals.index.to_list()
    split = np.array_split(ids, n_cores)

    chunks = []
    for l, i in zip(split, range(len(split))):
        start = l[0]
        end = l[-1]
        chunks.append((df_raw_signals.iloc[start : end + 1], i))

    # 使用多进程处理每个子集
    my_pool = Pool(n_cores)
    result = my_pool.map(sub_features, chunks)
    result = sorted(result, key=lambda tup: tup[0])

    # 合并处理后的数据集
    df_list = [item[1] for item in result]
    df_final = pd.concat(df_list)
    df_final = df_final.reset_index(drop=True)

    return df_final
