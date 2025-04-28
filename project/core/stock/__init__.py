# core/stock/__init__.py

# 导出关键功能（重要！）
from .tushare_data_provider import get_technical_factor

# 声明公开导出的函数/类
__all__ = ['get_technical_factor']

# 子包初始化代码（可选）
print("Stock subpackage initialized")