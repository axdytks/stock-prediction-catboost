import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from matplotlib import font_manager

# 指定系统已安装的中文字体（例如文泉驿微米黑）

# plt.rcParams["font.sans-serif"] = ["SimHei"]  # Windows系统
# plt.rcParams["font.sans-serif"] = ["PingFang TC"]       # macOS系统

# 解决负号显示问题
plt.rcParams["axes.unicode_minus"] = False


class PredictionVisualizer:
    """预测结果可视化工具类"""
    def __init__(self, output_dir: str = "./output/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_predictions_vs_actuals(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        title: str = "Actuall vs Predict",
        save_path: Optional[str] = None,
        dpi: int = 300,
    ) -> None:
        """绘制实际值与预测值的对比图"""
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(len(actuals)), y=actuals, label="Actual")
        sns.lineplot(x=range(len(predictions)),
                     y=predictions,
                     label="Predict",
                     linestyle="--")
        plt.title(title)
        plt.xlabel("TimeSeries")
        plt.ylabel("Price")
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        else:
            default_path = self.output_dir / "predictions_vs_actuals.png"
            plt.savefig(default_path, dpi=dpi, bbox_inches="tight")
        plt.close()

    def plot_feature_importance(
        self,
        model: CatBoostRegressor,
        feature_names: list,
        top_n: int = 10,
        title: str = "Feature_significance",
        save_path: Optional[str] = None,
        dpi: int = 300,
    ) -> None:
        """绘制特征重要性图"""
        importance = model.get_feature_importance()
        sorted_idx = np.argsort(importance)[-top_n:]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance[sorted_idx],
                    y=np.array(feature_names)[sorted_idx])
        plt.title(title)
        plt.xlabel("SigPoint")
        plt.ylabel("Feature")

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        else:
            default_path = self.output_dir / "feature_importance.png"
            plt.savefig(default_path, dpi=dpi, bbox_inches="tight")
        plt.close()

    def plot_residuals(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        title: str = "Residual_Distribution",
        save_path: Optional[str] = None,
        dpi: int = 300,
    ) -> None:
        """绘制残差分布图"""
        residuals = actuals - predictions

        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.title(title)
        plt.xlabel("Residual")
        plt.ylabel("Frequency")

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        else:
            default_path = self.output_dir / "residuals_distribution.png"
            plt.savefig(default_path, dpi=dpi, bbox_inches="tight")
        plt.close()

    def visualize_all(
        self,
        model: CatBoostRegressor,
        actuals: np.ndarray,
        predictions: np.ndarray,
        feature_names: list,
        prefix: str = "",
    ) -> None:
        """一键生成所有可视化图表"""
        # 实际 vs 预测
        self.plot_predictions_vs_actuals(
            actuals,
            predictions,
            title=f"{prefix} Actual price vs Predicted price",
            save_path=self.output_dir / f"{prefix}_predictions_vs_actuals.png")

        # 特征重要性
        self.plot_feature_importance(model,
                                     feature_names,
                                     title=f"{prefix} feature_importance",
                                     save_path=self.output_dir /
                                     f"{prefix}_feature_importance.png")

        # 残差分布
        self.plot_residuals(actuals,
                            predictions,
                            title=f"{prefix} Residuals_distribution",
                            save_path=self.output_dir /
                            f"{prefix}_residuals_distribution.png")


if __name__ == "__main__":
    # 示例用法
    visualizer = PredictionVisualizer()

    # 假设已有模型和测试数据
    dummy_actuals = np.random.rand(100) * 100
    dummy_predictions = dummy_actuals + np.random.normal(0, 5, 100)
    dummy_model = CatBoostRegressor()
    dummy_feature_names = [f"feature_{i}" for i in range(20)]

    # 生成所有图表
    visualizer.visualize_all(model=dummy_model,
                             actuals=dummy_actuals,
                             predictions=dummy_predictions,
                             feature_names=dummy_feature_names,
                             prefix="dummy")
