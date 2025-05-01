import pandas as pd
import tushare as ts
from datetime import datetime, timedelta
import os
import re
from tqdm import tqdm
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# ========================
# Tushare 配置
# ========================
# 需要先在 https://tushare.pro 注册获取token
TUSHARE_TOKEN = "284b804f2f919ea85cb7e6dfe617ff81f123c80b4cd3c4b13b35d736"  # 替换成你的实际token
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# ========================
# 数据缓存配置
# ========================
CACHE_DIR = "./data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_technical_factor(symbol: str,
                         days: int = 400,
                         end_date: Optional[str] = None,
                         use_cache: bool = True) -> pd.DataFrame:
    """
    获取股票技术指标数据（确保返回纯数值型数据）
    
    参数:
        symbol: 股票代码（支持A股代码，如 "601127.SH" 或简写 "601127"）
        days: 需要获取的数据天数
        end_date: 截止日期（格式：YYYYMMDD），默认取最近交易日
        use_cache: 是否使用本地缓存
    
    返回:
        pd.DataFrame: 包含技术指标的DataFrame，按日期倒序排列，仅含数值型列
    """
    # 标准化股票代码格式
    if not symbol.endswith(('.SH', '.SZ')):
        symbol = f"{symbol}.SH" if symbol.startswith(
            ('6', '9')) else f"{symbol}.SZ"

    # 生成缓存文件名
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{days}.pkl")

    # 尝试读取缓存
    if use_cache and os.path.exists(cache_file):
        df = pd.read_pickle(cache_file)
        print(f"Loaded cached data for {symbol} ({len(df)} records)")
        return df

    # 确定日期范围
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.strptime(end_date, "%Y%m%d") -
                  timedelta(days=days * 2)).strftime("%Y%m%d")

    # 获取基础数据
    try:
        df = pro.daily(ts_code=symbol,
                       start_date=start_date,
                       end_date=end_date)
        print("Tushare返回的原始列名:", df.columns.tolist())  # 新增调试

    except Exception as e:
        raise ValueError(f"Tushare API error: {str(e)}") from e

    if df.empty:
        raise ValueError(f"No data found for {symbol}")

    # 转换为datetime格式并排序
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.sort_values('trade_date', ascending=False).reset_index(drop=True)

    # ========================
    # 数据清洗（关键修改！）
    # ========================
    # 重命名列并删除非数值列
    df = df.rename(columns={
        'trade_date': 'date',
        'vol': 'volume'
    }).drop(columns=['ts_code'])

    # ========================
    # 计算技术指标
    # ========================
    # 移动平均线
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()

    # RSI（14日）
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))

    # MACD（12/26/9）
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # 布林带（20日）
    df['std'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['ma20'] + 2 * df['std']
    df['lower_band'] = df['ma20'] - 2 * df['std']

    # 处理NaN值（删除包含NaN的行）
    df = df.dropna()

    # 保留指定天数数据
    df = df.head(days)
    # 数据清洗步骤

    # 添加调试信息
    print("清洗后数据列:", df.columns.tolist())
    print("数据类型:\n", df.dtypes)

    # 保存缓存
    if use_cache:
        df.to_pickle(cache_file)
        print(f"Cached data saved for {symbol}")

    return df


