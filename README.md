# Kalman_Filter

> A Python implementation of Kalman Filter, reproduced from DR_CAN’s teaching workflow and Excel version for learning and verification purposes.

## 📌 Introduction

`Kalman_Filter` 是一个用于学习与实践卡尔曼滤波（Kalman Filter）的 Python 项目。  
本项目基于 DR_CAN 视频中的讲解思路，以及其提供的 Excel 版本计算流程，完成了对应的 Python 复现实现。

项目重点不在“封装成黑盒库”，而在于：

- 清晰呈现 **预测（Predict）- 更新（Update）** 两阶段流程
- 对照 Excel 版本逐步验证关键公式与中间变量
- 通过代码方式沉淀对卡尔曼滤波原理的理解

## 🎯 Objectives

- 理解卡尔曼滤波的核心数学流程
- 将表格化计算逻辑迁移到可复用的 Python 实现
- 通过可运行脚本观察滤波前后估计结果变化
- 为后续扩展到多维状态、非线性滤波（EKF/UKF）打基础

## 🧠 Algorithm Overview

标准离散卡尔曼滤波通常分为两步：

1. **Predict（预测）**
   - 状态预测  
     \[
     \hat{x}_{k|k-1} = A\hat{x}_{k-1|k-1} + Bu_k
     \]
   - 协方差预测  
     \[
     P_{k|k-1} = AP_{k-1|k-1}A^T + Q
     \]

2. **Update（更新）**
   - 卡尔曼增益  
     \[
     K_k = P_{k|k-1}H^T(HP_{k|k-1}H^T + R)^{-1}
     \]
   - 状态更新  
     \[
     \hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k(z_k - H\hat{x}_{k|k-1})
     \]
   - 协方差更新  
     \[
     P_{k|k} = (I - K_kH)P_{k|k-1}
     \]

## 🏗️ Project Structure

> 请按你的实际文件名调整（下面是推荐结构）

```text
Kalman_Filter/
├─ kalman_filter.py        # 卡尔曼滤波核心实现
├─ example.py              # 示例脚本（构造观测数据并运行滤波）
├─ requirements.txt        # 依赖列表（如 numpy / matplotlib）
└─ README.md
```

## ⚙️ Environment

- Python 3.8+
- 推荐依赖：
  - `numpy`
  - `matplotlib`（可选，用于可视化）

安装方式：

```bash
pip install -r requirements.txt
```

如果暂无 `requirements.txt`，可先手动安装：

```bash
pip install numpy matplotlib
```

## 🚀 Quick Start

```bash
git clone https://github.com/Boooooobbb/Kalman_Filter.git
cd Kalman_Filter
python example.py
```

运行后你通常可以看到：

- 原始观测值（含噪声）
- 滤波估计值（更平滑、更接近真实趋势）
- （可选）图像对比曲线

## 🧪 Reproducibility Notes

本项目强调与 DR_CAN 的 Excel 思路对照验证。建议你在调试时：

- 固定随机种子（便于复现实验）
- 打印每一步关键变量（如 \(K, P, \hat{x}\)）
- 将 Python 中间结果与 Excel 逐项对照

## 📚 Reference

- DR_CAN 教学视频：  
  https://www.bilibili.com/video/BV1dV411B7ME/?spm_id_from=333.1391.0.0&vd_source=2e103011b6d82764f44fa4d68c3a5d60

> 本仓库为个人学习复现项目，核心思路参考上述教学内容。

## 🛣️ Roadmap

- [ ] 增加 1D / 2D 多场景示例
- [ ] 增加参数敏感性实验（Q/R 对滤波效果影响）
- [ ] 增加可视化与实验报告
- [ ] 封装为可复用类接口
- [ ] 扩展到 EKF / UKF

## 🤝 Contributing

欢迎交流与改进建议。  
如果你对卡尔曼滤波建模、参数调优或工程化封装有想法，欢迎提 Issue / PR。

## 📄 License

建议使用 MIT License（如需我可以帮你补一个标准 `LICENSE` 文件）。
