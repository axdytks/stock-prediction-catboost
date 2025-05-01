# core/utils/__init__.py

# 导出 log.py 中的 logger（假设 log.py 中有 logger）
from .log import logger

# 声明公开导出的对象
__all__ = ['logger']

# 子包初始化代码（可选）
print("Utils subpackage initialized")