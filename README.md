# Monte Carlo Normalizing Constant Estimation

通过二维非线性目标，研究 Uniform Monte Carlo、Gaussian Importance Sampling 和建议分布设计对积分估计的影响。

当前实现了四种非 oracle 方法、精确采样对照，以及 raw → summary → comparison 的结果处理流程。每次重复估计都会保存，统计量与比较图使用同一批数据。重点是区分有限样本表现、截断偏差和重要性权重的尾部性质。

![四种方法的重复估计均值与经验 RMSE 比较](figures/comparison.png)

## 问题与真值

目标未归一化密度为

$$
\widetilde\rho(x_1,x_2)
=\exp\left[-\frac{x_1^2}{2}
-\frac{(2x_2+x_1^2-1)^2}{2}\right],
\qquad
Z=\int_{\mathbb R^2}\widetilde\rho(x)\,dx.
$$

高密度区域沿曲线 $x_2=(1-x_1^2)/2$ 分布。令 $u=x_1$、$v=2x_2+x_1^2-1$，逆变换的 Jacobian 绝对值为 $1/2$，所以

$$
Z=\frac12\int_{\mathbb R^2}e^{-(u^2+v^2)/2}\,du\,dv=\pi.
$$

Uniform MC 在 $[-4,4]^2$ 上估计的是截断积分：

$$
Z_{[-4,4]^2}\approx3.131708411797,
\qquad
Z-Z_{[-4,4]^2}\approx0.009884241793.
$$

它对截断积分无偏，但对完整积分有固定截断偏差。当前所有方法的经验 RMSE 都以完整真值 $\pi$ 计算，因此包含 Uniform 的截断误差。

## 当前方法与结果

| 方法 | 抽样分布 | raw 文件名 |
|---|---|---|
| Uniform MC | $U([-4,4]^2)$ | `uniform_truncated_4.csv` |
| Initial Gaussian IS | $N((0,0.5),\operatorname{diag}(1,0.25))$ | `gaussian.csv` |
| Refined Gaussian IS | $N((0,0),\operatorname{diag}(2.46006384, 1.40547353))$ | `refined_gaussian.csv` |
| Moment-matched Gaussian IS | $N((0,0),\operatorname{diag}(1,0.75))$ | `moment_matched_gaussian.csv` |

现有数据使用 $N\in\{100,300,1000,3000,10000,30000,100000\}$，每个 N 重复 100 次。以下为从当前 raw 重新汇总得到的经验 RMSE，均以完整真值 $\pi$ 为参照：

| N | Uniform MC | Initial Gaussian IS | Refined Gaussian IS | Moment-matched Gaussian IS |
|---:|---:|---:|---:|---:|
| 10,000 | 0.095375 | 0.907684 | 0.039548 | 0.104686 |
| 100,000 | 0.030459 | 1.258050 | 0.015139 | 0.061507 |

示例规模下，Refined Gaussian 的经验 RMSE 最低；匹配协方差的 Gaussian 仍有明显误差和波动。图中的幂律拟合仅描述这批结果，不用于证明理论收敛阶。

### 种子设定

Gaussian IS 的调参和四方法比较使用不同随机流，避免调参结果对比较图产生偏倚。调参过程使用 `seed=20261008`，四方法比较使用 `seed=20260930`。调参完成后，固定 Refined Gaussian 的参数。各方法分别初始化生成器，并连续生成重复估计所需的样本。

### 精确采样对照

本题可以直接生成归一化目标分布的独立样本：

$$
X_1\sim N(0,1),\qquad
X_2\mid X_1=x\sim N\left(\frac{1-x^2}{2},\frac14\right).
$$

实现位于 `experiments/exact.py`。对应密度 $q=\widetilde\rho/\pi$，重要性权重恒为 $\pi$，用于零方差 oracle 对照和实现自检。它利用本题的解析结构，不代表一般未知归一化常数问题的通用解法。

该入口只保存 raw，不调用 RMSE 绘图。summary 保留 exact 的全部统计量及真实零值；浮点舍入可能产生机器精度量级的差异。为避免对零 RMSE 取对数，exact 不纳入四方法比较图。

## Raw 与 summary 的统计口径

`results/raw/<method>.csv` 每行是一轮独立重复估计，包含 `Sample Size,Replicate,Estimate`。程序从文件名提取 method，按方法和 N 分组，输出 [results/summary/summary.csv](results/summary/summary.csv)。当前五种h核心方法共 3,500 条 raw 记录，汇总为 35 行。

| 字段 | 含义 |
|---|---|
| `method` | raw CSV 的文件名，不含扩展名 |
| `n_samples` | 每次估计的样本量 N |
| `n_replicates` | 当前方法和 N 对应的实际重复次数 R |
| `mean_estimate` | R 次估计的均值 |
| `bias_vs_pi` | 重复估计均值减去 $\pi$ |
| `empirical_variance` | 重复估计的样本方差，使用 `ddof=1`，除以 R−1 |
| `rmse_vs_pi` | $\sqrt{R^{-1}\sum_r(\widehat Z_r-\pi)^2}$ |
| `median_abs_error` | 每次估计绝对误差的中位数 |
| `q95_abs_error` | 每次估计绝对误差的 95% 分位数，使用线性插值 |
| `max_abs_error` | 每次估计绝对误差的最大值 |

所有误差都基于每次重复的估计值计算，而不是只对重复均值计算。只有一次重复时，样本方差未定义，CSV 中留空。程序会拒绝缺失值、非有限估计、非正整数 N/重复编号及重复的 `(Sample Size, Replicate)`，不会静默丢弃异常结果。

比较图从同一份 summary 读取 `mean_estimate` 和 `rmse_vs_pi`。各个实验入口的单方法估计图仍展示第一轮重复的估计值；它与比较图的重复均值是不同的统计量。每个 N 共生成 $RN$ 个样本，RMSE 使用这些已保存的重复估计，不再额外模拟。

## 环境与运行

主流程使用 Python、NumPy、SciPy、Matplotlib、pandas 和 tqdm。以下为 Windows PowerShell 命令，均在仓库根目录执行。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**从已有 raw 重建 summary 和比较图**，不重新采样：

```powershell
.\.venv\Scripts\python.exe -m experiments.export_summary
.\.venv\Scripts\python.exe -m experiments.comparison
```

输出分别为 `results/summary/summary.csv` 和 `figures/comparison.png`。比较图使用非交互绘图后端直接保存；更新 raw 后，应先重新导出 summary，再重建 comparison。

如需重新生成实验，依次运行以下命令。默认预算较大，会覆盖同名 raw CSV 和单方法图片：

```powershell
.\.venv\Scripts\python.exe -m experiments.uniform_truncated
.\.venv\Scripts\python.exe -m experiments.gaussian
.\.venv\Scripts\python.exe -m experiments.refined_gaussian
.\.venv\Scripts\python.exe -m experiments.matched_gaussian
.\.venv\Scripts\python.exe -m experiments.exact
.\.venv\Scripts\python.exe -m experiments.export_summary
.\.venv\Scripts\python.exe -m experiments.comparison
```

Gaussian 调参是独立入口，汇总已有结果不需要重跑它：

```powershell
.\.venv\Scripts\python.exe -m experiments.find_refined_is
```

该入口打印候选参数和优化器的终止状态，不会自动更新 `refined_gaussian.py` 中的固定参数。

## 代码与结果组织

| 位置 | 内容 |
|---|---|
| `src/target.py` | 目标密度、log-target、解析真值及截断数值积分 |
| `src/proposals.py` | 通用 proposal、Uniform 和 Gaussian 抽样与密度 |
| `src/estimators.py` | 通用 IS 估计、带可选进度条的重复估计、经验 RMSE 和 bias |
| `src/evaluations.py` | 根据估计值数组计算评价指标 |
| `src/experiments.py` | 遍历 N、保存每次重复估计并绘图 |
| `src/plotting.py` | 样本可视化、估计曲线、经验 RMSE 和拟合线 |
| `experiments/export_summary.py` | raw 分组汇总、统计定义及 summary CSV 导出 |
| `experiments/comparison.py` | 读取 summary，对比四种方法的均值与经验 RMSE |
| `experiments/` | 具体实验、调参、精确采样及结果处理入口 |
| `results/raw/` | 各方法的每次重复估计 |
| `results/summary/` | 各方法及样本量的统一统计汇总 |
| `figures/` | 单方法图与四方法比较图 |
| `notebooks/` | 基础模拟与方差缩减的学习实验 |
| `reports/` | LaTeX 报告源码及编译后的 PDF |
| `configs/` | 存储完整参数、seed、包版本等相关配置信息 |

raw 是统计来源，summary 和 comparison 都能从 raw 重建。完整参数、seed、包版本等相关配置信息存储在 `configs/` 中。

## 结论边界与下一步

- Gaussian IS 对本题完整积分无偏，但任意非退化二维 Gaussian proposal 的重要性权重二阶矩无限。有限重复得到的经验方差、RMSE 和拟合斜率不能证明有限理论方差或通常的 $N^{-1/2}$ RMSE 规律。
- Uniform 对完整积分的理论 RMSE 包含截断偏差；增大 N 不能消除约 `0.009884` 的误差下限。
- 匹配均值或协方差不能充分描述弯曲的目标分布。
- 当前结果尚不足以宣称某种方法在所有预算和随机种子下都更优，也未比较计算时间。

## License

[MIT](LICENSE)
