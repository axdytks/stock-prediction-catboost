# core/utils/log.py
import logging

# 初始化根日志器
logger = logging.getLogger("log")
logger.setLevel(logging.DEBUG)

# 控制台Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 文件Handler（可选）
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.ERROR)

# 统一格式化
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 添加Handler
logger.addHandler(console_handler)
logger.addHandler(file_handler)
