# core/__init__.py

# 声明公开导出的模块（可选）
__all__ = ['utils', 'stock']

# 包级别的初始化代码（可选）
print("Core package initialized")

# 显式导入子模块（可选，方便直接通过 core 访问）
from . import utils
from . import stock