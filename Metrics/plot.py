import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
def read_data(file_name, method):
    df = pd.read_excel(file_name, sheet_name="Sheet1")
    df = df[df["patient"].isin([f"CT{i}" for i in range(9)])]  # 筛选CT0-CT8
    df["Method"] = method
    return df[["patient", "sigma_res_mean", "NRR_mean", "AAG_ratio_mean", "L2_per_px_mean", "grad_cos_mean", "Method"]]

df_clip = read_data("ldct_pair_patient_summary_CLIP.xlsx", "CLIP")
df_cnn = read_data("ldct_pair_patient_summary_CNN10.xlsx", "CNN10")
df_core = read_data("ldct_pair_patient_summary_coreDiff.xlsx", "coreDiff")

# 合并数据
df_combined = pd.concat([df_clip, df_cnn, df_core])
df_mean = df_combined.groupby("Method")[["sigma_res_mean", "NRR_mean", "AAG_ratio_mean", "L2_per_px_mean", "grad_cos_mean"]].mean().reset_index()

# 定义指标和颜色
metrics = ["sigma_res_mean", "NRR_mean", "AAG_ratio_mean", "L2_per_px_mean", "grad_cos_mean"]
colors = {"CLIP": "blue", "CNN10": "orange", "coreDiff": "green"}
# 绘制图表
metrics = ["sigma_res_mean", "NRR_mean", "AAG_ratio_mean", "L2_per_px_mean", "grad_cos_mean"]
colors = {"CLIP": "blue", "CNN10": "orange", "coreDiff": "green"}

plt.figure(figsize=(15, 10))
for i, metric in enumerate(metrics, 1):
    plt.subplot(3, 2, i)
    for method in ["CLIP", "CNN10", "coreDiff"]:
        subset = df_combined[df_combined["Method"] == method]
        plt.plot(subset["patient"], subset[metric], label=method, color=colors[method], marker="o")
    plt.title(metric)
    plt.xlabel("Patient")
    plt.ylabel("Value")
    plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 8))
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 3, i)
    # 提取每个方法对应的颜色和数值
    methods = df_mean["Method"].tolist()
    values = df_mean[metric].tolist()
    bar_colors = [colors[method] for method in methods]
    # 绘制柱状图并添加数值标签
    bars = plt.bar(methods, values, color=bar_colors)
    plt.title(f"Mean {metric}")
    plt.xlabel("Method")
    plt.ylabel("Mean Value")
    # 在柱子上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()

from matplotlib.pyplot import figure
import numpy as np

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

for method in df_mean["Method"]:
    values = df_mean[df_mean["Method"] == method][metrics].values.flatten().tolist()
    values += values[:1]  # 闭合图形
    ax.plot(angles, values, label=method, color=colors[method], marker="o")
    ax.fill(angles, values, color=colors[method], alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_title("Model Performance Radar Chart")
plt.legend()
plt.show()