# Ratcliff DDM、LBA 与 RDM：软件初始化方法与参数边界比较

整理日期：2026-10-04。

本文记录 Ratcliff drift-diffusion model（DDM）、linear ballistic accumulator（LBA）与 racing diffusion model（RDM）相关软件的拟合初始化方法、Lower/Upper 设置依据，以及 PsySummary 当前采用的方案。它是设计说明与来源核对记录，不是参数估计结果。第 1–2 节说明通用概念及 DDM 参数，第 3–10 节覆盖 DDM，第 11 节覆盖 LBA，第 12 节覆盖 RDM，第 13–14 节提供三模型汇总及新增来源。

资料来自软件作者论文、官方文档及公开源码。不同版本和拟合接口可能采用不同规则，尤其是经典 HDDM 与 HDDMnn、HSSM 的不同似然、PyDDM 的不同优化器。下文不将某个示例推广为整个软件的统一默认行为，也不声称覆盖所有扩散模型实现。

## 1. 需要区分的五种数值

| 名称 | 含义 | 对拟合的影响 |
|---|---|---|
| 优化器初始值／Start | 优化开始时的一组参数 | 决定局部优化从哪里出发；多起点拟合可以使用多组 |
| 初始种群 | 全局优化开始时的多组候选参数 | 差分进化等方法通常没有唯一的单点初始值 |
| Lower／Upper | 参数搜索边界 | 限制整个优化过程；不是只限制第一步 |
| Fixed 值 | 不参与优化的参数值 | 所有候选解都使用该固定值 |
| 贝叶斯先验 | 对参数的概率分布假设 | 参与后验推断；先验均值不等于链初值，先验也未必有有限上界 |

“Upper 的初始值”在本文中特指：模型设置界面第一次自动填入 Upper 栏的默认搜索上界。它与“随机起点分布的上限”不同。

例如，随机生成 `t0 ~ U(0, 0.5)` 并不意味着优化时必须满足 `t0 <= 0.5`。只有优化器还设置了 Upper=0.5，后续搜索才受这个边界限制。

## 2. 参数定义与跨软件换算

| 参数 | 本文及 PsySummary 的含义 | 比较时需要注意 |
|---|---|---|
| `a` | 两个吸收边界之间的距离 | HSSM 的 `a` 使用半边界间距尺度；PyDDM 常见对称边界组件使用边界距离 `B`，全间距为 `2B` |
| `v` | 平均漂移率 | 与扩散噪声尺度 `s` 联动；不同 `s` 下的数值不能直接比较 |
| `zr` | 相对证据起点，`zr=z/a` | 无偏向为 `zr=0.5`；这里的证据起点不同于优化器 Start |
| `sz` | 证据起点均匀分布的绝对总宽度 | fast-dm 采用相对宽度 `szr=sz/a`，不能直接照搬数值 |
| `t0` | PsySummary 中非决策时间均匀分布的下端点 | 其他软件可能用同名参数表示平均非决策时间 |
| `st0` | 非决策时间跨试次均匀分布的总宽度 | 不是标准差；均匀分布的标准差为 `st0/sqrt(12)` |
| `sv` | 漂移率跨试次正态变异的标准差 | 不是区间宽度 |
| `d` | 下／上边界之间的非决策时间差异 | 必须核对符号和编码；不是所有实现都提供该参数 |
| `s` | 试次内扩散噪声尺度 | 为识别模型，通常固定一个尺度参数；PsySummary 默认固定 `s=1` |

时间统一用秒。毫秒数据进入拟合时应按配置转换，不能将 800 ms 直接写成参数值 800。

PsySummary 当前数值实现的非决策时间为：

```text
lower 响应：U(t0 + d/2, t0 + d/2 + st0)
upper 响应：U(t0 - d/2, t0 - d/2 + st0)
```

因此 `d=0` 时，平均非决策时间为 `t0 + st0/2`。当 `st0>0` 时，不能把拟合得到的 `t0` 直接解释成平均非决策时间。

fast-dm 教程使用的是以 `t0` 为均值的均匀分布 `U(t0-st0/2, t0+st0/2)`。忽略响应偏移时，两种参数化的换算为：

```text
t0_mean = t0_lower + st0/2
t0_lower = t0_mean - st0/2
```

HSSM 与 HDDM 的边界尺度差异见 [HSSM 官方说明](https://lnccbrown.github.io/HSSM/explanations/coming_from_hddm/)；fast-dm 的时间分布和相对宽度定义见 [作者教程](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00336/full)。其他软件仍应逐项核对其实际分布定义。

## 3. 软件初始化方法总表

| 软件／接口 | 初始值如何生成 | 数据依赖 | 核实范围及限制 |
|---|---|---|---|
| fast-dm 30／3.0 | 初始单纯形的第一组参数通过 EZ 生成；`zr=0.5`，变异参数初始为零；其余顶点通过小幅增加某个参数产生 | EZ 核心估计依赖正确率及正确 RT 的均值、方差 | 变异参数初始为零不代表固定为零；不将此规则未经核对推广到旧版本 [S1] |
| DMAT | `Guess=[]` 时自动生成；`GuessMethodScalar=1` 使用 EZ，设为 2 对 EZ 结果做小幅扰动；允许用户提供条件×参数矩阵 | 是 | 已核实 EZ／扰动选项；完整扩展参数的逐项补充值尚未全部核实 [S2] |
| fddm：`ddm()` | 核心参数截距 `v/a/t0` 使用 EZ；偏向 `w=0.5`，`sv=0`，差异系数为零；允许 `args_optim$init` 覆盖 | 是 | 针对该拟合接口；不是密度函数自身的初始化规则，不覆盖全部 Ratcliff 变异参数 [S3–S4] |
| rtdists 官方 vignette | 从经验范围随机抽取，检查初始目标函数是否可用，不可用则重新抽取，再调用 `nlminb` | 抽取分布不依赖数据，可用性筛选依赖数据 | 官方示例，不是包内统一的拟合默认接口 [S5] |
| PyDDM：默认差分进化 | 在用户边界内初始化候选种群；未覆盖选项时使用 SciPy 默认 Latin hypercube 初始化 | 不采用 EZ 汇总估计 | 初始化对象是种群；不能把局部优化的单点生成规则当成这一默认流程 [S6–S8] |
| PyDDM：局部优化 | 使用用户指定的 `Fittable` 默认值；未指定且双侧边界有限时，以 `L+(U-L)*Beta(2,2)` 生成单点 | 否 | 倾向区间中部；单侧或无边界时另有规则 [S7] |
| 经典 HDDM | 参数节点先有数值初值；调用 `find_starting_values()` 可通过 MAP／近似 MAP 优化获得采样起点 | 节点基础值不依赖数据，MAP 依赖数据 | 不是默认 EZ；层级节点、先验均值和最终采样起点不同 [S9–S10] |
| HSSM | 模型配置及回归结构对应的初始设置，可覆盖并加入 jitter；链接函数改变系数与实际参数的关系 | 教程基础值不是 EZ 估计 | 简单 DDM 教程数值不能推广到所有似然和层级结构 [S11–S12] |
| DMC／ggdmc | 已核实的 ggdmc 接口从参数先验生成起点；可用 `start.prior` 单独控制，也可给 `theta1` 矩阵 | 抽取阶段不直接依赖 RT 汇总统计 | 核实对象为所链接的 TasCL/ggdmc 实现，不保证所有 DMC 版本相同 [S13] |
| EMC2 | `init_chains()` 支持从用户定义的多元正态生成多个起点，由 `start_mu/start_var` 控制 | 该接口不是 EZ 方法 | 此处核实的是可配置接口；未把其示例均值认定为所有模型的自动默认值 [S14] |
| brms／Stan Wiener | 默认在无约束空间抽取 `U(-2,2)` 后变换到实际参数空间；支持用户初值 | 否 | 不是在实际 `a/v/t0/zr` 上全部抽取 `U(-2,2)`；标准 Wiener 家族不同于完整变异参数模型 [S15–S16] |
| CHaRTr | 使用 DEoptim 差分进化，初始化一组候选解；通过边界和优化器配置控制 | 未核实到统一 EZ 规则 | 需要比较种群配置，不能只寻找一组单点初值 [S17] |
| RWiener：`wdm()` | 接受 `start=c(alpha,tau,beta,delta)` | 自动规则尚未核实 | 接口支持四参数拟合；未确认 `start=NULL` 时的内部数值算法 [S18] |
| dRiftDM | 提供 EZ 与 Latin hypercube 搜索选项；接受一组或多组 `start_vals` | 启用 EZ 时是 | 方法依赖组件、拟合方式和优化器，不是所有模型共同的一套默认值 [S19] |

## 4. EZ 的角色：生成点估计，不生成 Upper

EZ-diffusion 是简化模型的闭式估计方法。典型输入是正确率、正确反应 RT 均值和方差；输出核心参数 `v/a/Ter`。其假设包括无起点偏向以及不估计完整跨试次变异结构。

fast-dm、DMAT 和 fddm 可以借此为更复杂的优化提供起点。但 EZ 本身不计算参数置信区间或统一的优化 Upper，也不意味着 `Ter` 的估计值应该成为搜索上界。边界仍由模型支持、软件设计或用户设置决定。

数据结构特殊、正确率极端或 EZ 结果不可行时，需要软件各自的修正／回退策略。fddm 源码还会检查 EZ 得到的非决策时间：负值回退到最短 RT 的 1%，达到或超过最短 RT 时改为其 99%。这属于初始点修正，不是改变 Upper 的估计方法 [S4]。

## 5. rtdists 官方示例的逐参数随机规则

`U(l,u)` 表示均匀分布；`N(mu,sigma²)` 表示正态分布。

| 参数 | 示例生成规则 | 说明 |
|---|---|---|
| `a` | `U(0.5,3)` | 有多个边界条件时分别生成 |
| `v` | 标准正态 `N(0,1)` | 示例有多个有序刺激强度，对生成的漂移率排序；该排序不应推广到没有顺序的条件 |
| `zr` | `U(0.4,0.6)` | 示例优化变量名为 `z`，密度调用前乘 `a`，故在这里按相对起点列出 |
| `t0` | `U(0,0.5)` | 0.5 是抽样范围上限，不是搜索 Upper |
| `sz` | `U(0,0.5)` | 绝对总宽度 |
| `sv` | `U(0,0.5)` | 标准差 |
| `st0` | `U(0,0.5)` | 非决策时间总宽度 |
| `d` | `N(0,0.05²)` | 标准差为 0.05；只有模型实际选择该参数时才进入拟合 |

示例的 `get_start()` 提供可选参数集合，实际拟合仅选取相关自由参数。`ensure_fit()` 反复抽取，直到初始目标函数不再返回“不可能参数”的惩罚值，然后开始优化。示例调用 `nlminb` 时设置 Lower，未指定有限 Upper [S5]。

所以，应分别陈述以下两件事：

1. 使用相同／类似的随机参数分布。
2. 复现示例的完整初始化筛选与优化流程。

PsySummary 当前采用第 1 项的启发式规则，并采用自己的边界、联合约束和时间支持检查；不应声称完全复现 rtdists 拟合流程。

## 6. `t0 Upper` 的跨软件比较

| 软件／接口 | 已核实的上界做法 | 是否能直接作为 PsySummary 的默认值依据 |
|---|---|---|
| fddm：普通非决策时间截距 | 源码 `u_bds['ndt']=min(rt)`；用户可以提供其他边界 | 可参考“基于数据支持”的原则，但不能直接推广到均值参数化、有时间变异或复杂回归的所有情况 [S4] |
| rtdists vignette | 未传有限 Upper；靠目标函数惩罚不可能参数 | 不能据 `U(0,0.5)` 推导 Upper=0.5 [S5] |
| fast-dm 30 | 教程给 `t0` 典型范围 0.2–1.0 秒 | 1.0 是典型范围上端，不是本次已核实的硬搜索边界 [S1] |
| DMAT | 本次未确认具体版本自动生成 `Ter Upper` 的完整规则 | 暂不能将 0.8、1.0 或最短 RT 归为已核实的 DMAT 默认规则 [S2] |
| 经典 HDDM 信息先验 | `t` 使用正值 Gamma 家族，没有有限先验上界 | 先验均值 0.4 不等于 Upper；不能直接移植 [S9] |
| 经典 HDDM 非信息先验 | 源码给 `t` 非常宽的上界 `1e3` 秒 | 技术范围，不是适合界面照搬的认知时间推荐 [S9] |
| HSSM 解析／blackbox DDM | 参数范围为正值、无有限上界 | 仍需似然支持；不等于任意参数都可解释观测 [S12] |
| HSSM 神经近似 DDM | 配置 `t` 范围为 0–2.0 秒 | 与近似似然配置／适用域有关，不是普遍推荐 [S12] |
| PyDDM | 通过用户的 `Fittable` 和具体非决策组件设置 | 无适用于所有组件的统一有限 Upper [S7] |
| ggdmc／EMC2 | 由模型参数化、先验、截断／变换及用户配置决定 | 未核实到一个通用的 DDM `t0 Upper` 数字 [S13–S14] |
| brms／Stan Wiener | 由生成模型中的参数约束、链接和先验决定 | 默认初始化空间 `(-2,2)` 不是实际时间上界 [S15–S16] |
| CHaRTr | 搜索范围由模型／用户配置提供 | 不把某项研究采用的边界称为全部 DDM 的官方推荐 [S17] |
| RWiener／dRiftDM | 接口和优化器允许用户控制；本文未确认统一自动 `t0 Upper` 数字 | 保留未核实标记，不从模拟示例推定 [S18–S19] |

fddm 的文档文字与源码存在值得记录的差别：手册的默认上界段落写 `t0 < Inf`，但源码实际生成普通非决策时间截距上界时使用 `min(rt)`，手册示例输出也显示有限上界。因此本表采用源码的具体规则；复杂回归的差异系数仍可能使用无限界，不能将单个截距规则推广到所有系数 [S3–S4]。

## 7. 最短 RT、时间变异与联合约束

在没有污染反应／lapse 成分、要求每条保留观测有正似然的连续 DDM 中，非决策时间分布必须能覆盖观测的时间支持。以下为分布定义的数学推导，不是某软件统一的界面默认值。

| 参数化，暂令 `d=0` | 非决策时间支持 | 必要的时间支持条件 |
|---|---|---|
| `st0=0` | 固定 `t0` | `t0 < min(RT)` |
| `t0` 为均值 | `U(t0-st0/2,t0+st0/2)` | `t0-st0/2 < min(RT)`，并保证非决策时间不为负 |
| `t0` 为下端点 | `U(t0,t0+st0)` | `t0 < min(RT)` |

均值参数化中，`st0` Free 后 `t0` 可以超过某些最短 RT，因为分布下端点仍可能更小。下端点参数化中，增大 `st0` 不会降低下端点，不能靠它补救过大的 `t0`。

这里不要求 `t0+st0 < min(RT)`：这会错误要求所有可能的非决策时间都比最快观测更短。时间支持成立也不代表初始似然数值一定稳定。

PsySummary 还有边界响应偏移 `d` 和固定节点数值积分。实际初始检查按响应侧的最短 RT 判断，并检查最早积分节点是否仍早于观测；因此会比纯连续支持更严格。不能将 GUI Upper 的单个数字当作完整的可行性判断。

最短 RT 对极快异常值敏感，例如未正确设置单位、误录或提前反应。改用较高分位数可以改善启发式起点对极端值的敏感性，但若异常值仍留在纯 DDM 的拟合数据中，不能用分位数替代真正的观测支持约束。是否保留／过滤这些反应仍由用户决定。

## 8. PsySummary 当前采用的 Ratcliff 设置

本节通过 2026-10-04 的本地源代码核对，主要依据 `psyData/cognitiveModelSpec.py` 与 `psyData/cognitiveModels.py`。以下是程序当前行为，不是其他软件的推荐表。

### 8.1 默认模式与搜索边界

| 参数 | 默认模式 | 初始显示 Value 的基础／回退值 | Lower | Upper |
|---|---|---|---|---|
| `a` | Free | 1.0 | 0.05 | 5.0 |
| `v` | Free | 1.0 | -10.0 | 10.0 |
| `t0` | Free | `min(0.1,minRT/2)` | 0 | 0.8；与 `st0` 都 Free 时自动默认改为 0.5 |
| `zr` | Fixed | 0.5 | 0.001 | 0.999 |
| `d` | Fixed | 0 | -0.5 | 0.5 |
| `sz` | Fixed | 0 | 0 | 4.999 |
| `sv` | Fixed | 0 | 0 | 5.0 |
| `st0` | Fixed | 0 | 0 | 0.6 |
| `s` | Fixed | 1.0 | 0.001 | 10.0 |

基础／回退 Value 不是实际自动拟合起点：自动 Value 可以被界面预览替换，正式拟合按各组当前过滤后的数据重新生成。Fixed 参数使用保存的 Value，Lower/Upper 不使它参与优化。`s=1` 默认用于尺度识别；随机生成器未为自由 `s` 定义额外参考分布。

`sz Upper=4.999` 只是宽泛的独立搜索界，并不意味着任意 `a/zr` 都允许这样大的宽度；实际仍要求 `a*zr±sz/2` 严格位于 `(0,a)`。

### 8.2 自动随机起点

| 参数 | 当前自动起点分布（参数为 Free 且自动管理时） | 与 rtdists 示例的关系 |
|---|---|---|
| `a` | `U(0.5,3)` | 同参考范围 |
| `v` | `N(0,1)` | 同单个漂移抽样；不采用有序刺激条件的漂移排序 |
| `zr` | `U(0.4,0.6)` | 同相对起点范围 |
| `t0` | `U(0,min(0.5,minRT))` | 加入各拟合组当前保留 RT 的支持限制 |
| `sz` | `U(0,0.5)` | 同参考范围，另查联合约束 |
| `sv` | `U(0,0.5)` | 同参考范围 |
| `st0` | `U(0,0.5)` | 同参考范围，另查时间积分支持 |
| `d` | `N(0,0.05²)` | 同参考分布，另查边界时间偏移 |

生成时先将参考均匀范围与用户 Lower/Upper 取交集；若没有有效交集，则尊重用户范围，在用户范围内生成。正态抽样会在边界内重试，当前最多 32 次，未抽到则回退到用户允许区间内的均匀抽样。因此“rtdists-inspired”比“完全匹配 rtdists”更准确。

### 8.3 `t0/st0 Upper` 的明确设计决定

| 模式 | 自动默认 `t0 Upper` | 自动默认 `st0 Upper` |
|---|---|---|
| 只有 `t0` Free，`st0` Fixed | 0.8 s | 0.6 s（Fixed 时不参与搜索） |
| `t0` 与 `st0` 都 Free | 0.5 s | 0.6 s |

这些是 PsySummary 自己的默认搜索范围，不是 EZ 估计，不是 rtdists 官方硬边界，也不是 `minRT` 的稳健统计估计。

用户可以修改。只有仍由程序自动管理的时间 Upper 才随 Fixed/Free 模式切换更新；手动 Upper 不覆盖。不施加 `t0+st0/2<=0.8` 等额外平均时间上限。采用当前下端点参数化时，0.5/0.6 组合允许平均非决策时间达到 0.8 秒，但这不是额外联合约束。

### 8.4 Filters、手动值与多起点

1. 每次 Run 使用当前 Filters 后的数据；按实际 Rows/Columns 拟合组计算保留的有效 RT 最小值，并生成该组起点。
2. 随机生成使用分析运行保存的实际 seed；时间型 seed 在分析副本上解析，不能仅把 GUI 的“Time-based”草稿当作可重现数值。
3. 第一组候选点保留手动 Free Value；自动 Free 参数可以重抽。Fixed 值始终保留。
4. 后续候选点允许随机改变 Free 参数，包括手动指定的 Free 起点；“手动起点保护”不表示所有多起点都使用同一手动值。
5. 候选点必须满足独立边界、联合约束和时间支持。没有可行候选点时报告错误，不修改用户数据或过滤器。
6. 当前不设独立的初始似然预检查；正式优化过程中仍需计算目标函数。
7. Save Settings 和 Run 对保留 RT 小于 50 ms 的情况警告，允许继续；不强制过滤，不自动修改范围。

上述第 6 项是与 rtdists 示例完整流程的明确差异：我们保留便宜的可行性检查，不额外先计算一次完整初始似然。

### 8.5 证据起点与响应编码

界面和拟合结果使用 `zr`，每次密度调用转换为 `z=a*zr`。默认 Fixed=0.5 在 `a` 改变时仍保持无偏向。`sz` 保持绝对宽度。

Accuracy Coding 在正确物理响应为 lower 时将 `zr` 镜像为 `1-zr`，并将 `d` 取反；Response Coding 保持物理响应边界，并按当前实现反转相关试次的 `v`。两种编码均在每个拟合组内估计一套共享参数，不是分别拟合正确和错误反应。本文不提供旧绝对 `z` 设置的转换方案。

## 9. 文档、tooltip 和论文描述建议

建议使用：

> 自动初始化参考 rtdists 官方拟合示例的随机参数范围，并结合当前过滤后各拟合组的数据、用户搜索边界及联合约束生成。搜索边界由 PsySummary 默认设置或用户指定，独立于初始化分布。当前不进行单独的初始似然预检查。

`t0` 的说明需要明确：

> t0 是非决策时间分布的下端点。当 st0>0 时，忽略响应侧偏移的平均非决策时间为 t0+st0/2。Upper 是可修改的搜索上界，不是估计得到的平均非决策时间上限。

避免以下表达：

- “rtdists 推荐 t0 Upper=0.5”：示例的 0.5 是随机抽样上限。
- “EZ 计算出了 t0 Upper”：EZ 生成核心参数点估计。
- “fast-dm 的硬上界为 1 秒”：本次来源只核实到典型范围。
- “HSSM 推荐所有 DDM 的 t0 Upper=2 秒”：有限范围属于具体近似似然配置。
- “所有软件的 t0 都表示平均非决策时间”：不同实现的分布参数化不同。
- “按分位数设置范围可以消除保留极快 RT 的似然问题”：启发式起点与观测支持是不同问题。

## 10. 来源清单与后续核实项目

以下链接于整理时用于核对。GitHub 的 master/main 为可变来源；正式论文或发布时应固定版本／提交，并核对实际安装环境。

| 编号 | 来源 | 主要支持内容 |
|---|---|---|
| S1 | [Voss 等：fast-dm-30 教程（2015）](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2015.00336/full) | EZ 单纯形初始化、参数定义、典型范围 |
| S2 | [Vandekerckhove & Tuerlinckx：DMAT 指南（2008）](https://www.ppw.kuleuven.be/okp/_pdf/Vandekerckhove2008DMAWM.pdf) | Guess、GuessMethodScalar、短／长单纯形流程 |
| S3 | [fddm 官方参考手册](https://cran.r-universe.dev/fddm/doc/manual.html) | ddm 初始值、系数、用户覆盖接口 |
| S4 | [fddm：R/ddm.R 源码](https://raw.githubusercontent.com/cran/fddm/master/R/ddm.R) | min(rt) 时间截距上界、EZ 不可行起点修正 |
| S5 | [rtdists：Ratcliff & Rouder 1998 重分析源码](https://rdrr.io/cran/rtdists/f/inst/doc/reanalysis_rr98.Rmd) | 随机分布、相对起点转换、ensure_fit、未传有限 Upper |
| S6 | [PyDDM：functions.py](https://raw.githubusercontent.com/mwshinn/PyDDM/master/pyddm/functions.py) | 默认差分进化及其他优化路径 |
| S7 | [PyDDM：model 源码文档](https://pyddm.readthedocs.io/en/stable/_modules/pyddm/model.html) | Fittable.default 与用户边界 |
| S8 | [SciPy：differential_evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html) | 默认 Latin hypercube 种群初始化 |
| S9 | [HDDM：hddm_info.py](https://github.com/hddm-devs/hddm/blob/master/hddm/models/hddm_info.py) | 信息／非信息先验、节点初值及范围 |
| S10 | [Kabuki：hierarchical.py](https://github.com/hddm-devs/kabuki/blob/master/kabuki/hierarchical.py) | find_starting_values、MAP／近似 MAP |
| S11 | [HSSM：初始值教程](https://lnccbrown.github.io/HSSM/tutorials/initial_values/) | 简单 DDM 初值、jitter、链接函数影响 |
| S12 | [HSSM：DDM 配置源码](https://raw.githubusercontent.com/lnccbrown/HSSM/main/src/hssm/modelconfig/ddm_config.py)及[官方配置示例](https://lnccbrown.github.io/HSSM/archive/hssm_tutorial_workshop_1/) | 不同似然的时间参数范围 |
| S13 | [TasCL/ggdmc：samples.dmc](https://rdrr.io/github/TasCL/ggdmc/man/samples.dmc.html) | p.prior、start.prior、theta1 初始化 |
| S14 | [EMC2 官方参考手册](https://ampl-psych.r-universe.dev/EMC2/doc/manual.html) | init_chains、用户均值和协方差 |
| S15 | [brms：brm 文档](https://paulbuerkner.com/brms/reference/brm.html) | init 接口及 Stan 默认初始化 |
| S16 | [Stan：MCMC 初始化说明](https://mc-stan.org/docs/reference-manual/mcmc.html) | 无约束空间随机初始化与反变换 |
| S17 | [CHaRTr 作者页面](https://sites.bu.edu/chandlab/chartr/)及[作者论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6980795/) | DEoptim 全局拟合框架 |
| S18 | [RWiener：wdm 官方文档](https://search.r-project.org/CRAN/refmans/RWiener/html/wdm.html) | 用户 start、四参数拟合接口 |
| S19 | [dRiftDM 官方参考手册](https://cran.r-universe.dev/dRiftDM/doc/manual.html) | EZ／Latin hypercube 选项、手动多起点 |

后续若需要更强的逐版本比较，应补充：DMAT 完整自动边界与扩展参数初始化源码；RWiener 的自动 start 内部规则；EMC2 不传 start_mu/start_var 时的自动初始化；经典 DMC 不同版本；各框架特定 DDM 模型的完整边界配置。以上项目在本文中保留“未核实”或“依接口而定”，不补写未经确认的推荐数字。

## 11. LBA：初始化与边界比较

### 11.1 模型参数定义

LBA 是多个线性累积器的竞赛，每个响应对应一个累积器。试次内给定起点和漂移后，证据线性上升；跨试次随机起点及漂移使反应时变化。PsySummary 使用正态漂移的正漂移版本，并允许非决策时间变异。不同软件的条件化／截断约定仍需核对，不能仅因名称相同就认为似然完全相同。

| 参数 | PsySummary 含义 | 比较中的常见差异 |
|---|---|---|
| `A` | 每试次证据起点从 `U(0,A)` 抽取，平均起点为 `A/2` | 这不是优化器起点；也不同于 DDM 的相对起点 `zr` |
| `b` | 绝对反应阈值，必须满足 `b>A` | EMC2／部分 DMC 使用 `B=b-A`，并实际搜索正的阈值间隙 |
| `mean_v[i]` | 第 i 个累积器的跨试次漂移均值 | 某些实现允许负的正态均值；PsySummary 当前默认搜索界为正，不代表所有 LBA 都强制均值为正 |
| `sd_v[i]` | 第 i 个累积器漂移的跨试次标准差 | 与 RDM 的试次内扩散噪声不同；至少固定一个尺度参数才能识别模型 |
| `t0` | 非决策时间下端点 | 当前 `st0=0` 时也等于该固定时间 |
| `st0` | 非决策时间总宽度 | 非决策时间为 `U(t0,t0+st0)`，均值 `t0+st0/2` |

原始模型来源：[Brown & Heathcote（2008）](https://doi.org/10.1016/j.cogpsych.2007.12.002)。参数接口见 rtdists LBA 文档 [L3]；阈值间隙和尺度约定见 EMC2 LBA 文档 [L4]。

### 11.2 软件初始化方法与支持范围

| 软件／接口 | 初始化方法 | Lower／Upper 的来源 | `t0/st0` 相关限制 |
|---|---|---|---|
| rtdists LBA 帮助页示例 | `A~U(0,1)`，`b=A+U(0,1)`，`t0~U(0,minRT)`；自由漂移均值和第二个标准差 `U(0,1)`；第一个标准差固定 1 | 示例只给 `lower=0`，没有有限 Upper；`b>A` 先在起点构造中满足 | 不估计 `st0`；建议以不同随机起点重复运行，未提供 DDM vignette 那种统一 ensure_fit 重抽循环 [L1] |
| rtdists README 示例 | `A/b/t0~U(0,0.5)`；两个均值和第二个标准差 `U(0.5,2)`；第一个标准差固定 1 | 只设置零 Lower；没有有限 Upper | 初值规则与帮助页示例不同，进一步说明不存在唯一“rtdists LBA 默认拟合方案” [L2] |
| rtdists RR98 重分析示例 | 采用特定任务的随机参数生成器、条件漂移参数化及目标函数检查 | 来自该任务的参数化及显式设置，不是所有 LBA 的通用规则 | 可见其他随机范围，不能把研究示例中的模拟／条件约束直接移植为通用默认 [S5] |
| EMC2 LBA | 使用共享贝叶斯链初始化框架；`init_chains` 可按用户指定多元正态生成候选点 | `A/B/t0/sv` 为 log 变换，天然非负；漂移均值在实数轴；先验和用户配置控制实际推断区域 | 文档模型参数表未提供 `st0`；模型遗漏参数的固定默认值不同于自由参数链初值 [L4、S14] |
| DMC／ggdmc LBA | 从先验／起点先验生成链，也可提供显式参数初始矩阵；作者示例支持三累积器 LBA | `BuildPrior` 等指定截断范围，常用阈值间隙 `B` 保证可行性 | 作者示例固定 `sd_v=1、st0=0`；示例不是所有 LBA 的硬要求 [L5、S13] |
| HSSM `lba2/lba3/lba4` | 采用 HSSM／PyMC 初始化框架，可设置 initvals；未在本文确认各参数独立、固定的通用数值初值 | 当前源码：`A>=0`，`b>=0.2`，各 `v>=0`，Upper 均为无限；另要求 `b>A` | 已核实的这些内置似然参数列表没有独立 `t0/st0/sd_v`；不能直接与 PsySummary 的完整参数表逐项对齐 [L6–L7] |
| 自定义 Stan LBA | 由作者模型代码和用户 init 控制；使用 Stan 默认时在无约束空间随机生成 | 取决于模型代码、参数变换和先验，没有平台统一的 LBA 上界 | 必须查看具体实现，不能将 brms 标准 Wiener 家族当作现成 LBA 接口 [L8、S16] |

fast-dm、DMAT、经典 HDDM、fddm、RWiener 在本次已核实接口中主要针对 DDM，不在这里虚构其 LBA 初始化规则。对于新增、扩展或自定义模型，应单独列出明确实现及版本。

### 11.3 rtdists 帮助页示例与 PsySummary 的逐参数关系

| 参数 | rtdists LBA 帮助页示例 | PsySummary 当前自动起点 |
|---|---|---|
| `A` | `U(0,1)` | 同参考范围，与用户界取交集 |
| `b` | 先抽一个 `U(0,1)` 再加 `A` | `U(A,A+1)`，与用户界取交集，并检查 `b>A` |
| `t0` | `U(0,minRT)` | 同参考规则，minRT 来自当前过滤后的实际拟合组 |
| `mean_v[i]` | `U(0,1)` | 同参考范围；默认 Lower=0.01，故正常情况下实际交集为 `[0.01,1]` |
| `sd_v[1]` | Fixed=1 | 默认 Fixed=1；尊重用户保存的 Fixed 值 |
| 其他自由 `sd_v[i]` | 示例第二个标准差 `U(0,1)` | 对各自由标准差使用该范围，与用户界取交集 |
| `st0` | 未估计，默认 0 | 自由时 `U(0,min(0.5,minRT))`；这是 PsySummary 的扩展规则 |

这里的阈值间隙是随机初始化的构造方式，不表示整个拟合必须满足 `b-A<=1`。当前 GUI `b Upper=10`；优化器在用户边界和联合约束内搜索。

### 11.4 PsySummary LBA 的完整默认参数表

第 i 个响应对应第 i 个累积器，i 从 1 开始。

| 参数 | 默认模式 | 基础／回退 Value | Lower | Upper |
|---|---|---|---|---|
| `A` | Free | 0.5 | 0 | 5 |
| `b` | Free | 1.0 | 0.01 | 10 |
| `t0` | Free | `min(0.1,minRT/2)` | 0 | 0.8；与 `st0` 都 Free 时自动默认 0.5 |
| `st0` | Fixed | 0 | 0 | 0.6 |
| `mean_v[i]` | Free | `1.5-0.2*(i-1)` | 0.01 | 10 |
| `sd_v[1]` | Fixed | 1.0 | 0.01 | 5 |
| `sd_v[i>1]` | Free | 1.0 | 0.01 | 5 |

这些基础值只记录当前生成函数；正式自动拟合起点按第 11.3 节重新抽取，不能用基础 Value 表替代实际初始候选参数记录。高累积器数下基础漂移回退公式也不等于已经过可行性验证的正式起点。

当前 `t0 Upper` 的 0.8／0.5 与 `st0 Upper=0.6` 是产品默认边界，不由 rtdists 的 minRT 抽样规则计算。用户修改的 Upper 保留。固定标准差的值影响证据尺度，比较 `A/b/mean_v` 时必须同时核对标准差的识别约定。

### 11.5 LBA 时间边界与联合约束

- `A>=0`、`b>A`；所有自由漂移标准差必须正。
- 非决策时间为下端点参数化，连续支持要求 `t0<minRT`，而不是 `t0+st0<minRT`。
- PsySummary 初始时间检查还要求 `t0+c7*st0<minRT`，其中 `c7` 是 7 点 Gauss–Legendre 积分映射到 `[0,1]` 后最早节点的位置。此条件针对当前固定节点数值实现，不是所有 LBA 的理论约束。
- 至少固定一个 `sd_v` 用于尺度识别；默认固定第一个为 1。不能同时任意缩放证据阈值、均值和所有标准差而声称得到不同的可识别模型。
- 这些是拟合可行性检查，不自动改变 GUI Upper，也不自动过滤极短反应。

## 12. RDM：初始化与边界比较

### 12.1 模型定义及与 LBA／DDM 的差别

本文 RDM 指 racing diffusion model：每个响应有一个单边界扩散累积器，各累积器竞赛，最先到达自身阈值者决定响应。它不是“多个条件分别拟合 DDM”，也不是单个两边界 Ratcliff 模型。

| 参数 | PsySummary 含义 | 与其他模型的区别 |
|---|---|---|
| `A` | 起点从 `U(0,A)` 抽取 | 与 LBA 起点范围对应；不是 DDM 的 `zr` |
| `b` | 绝对阈值，`b>A` | EMC2 用正的间隙 `B=b-A` |
| `v[i]` | 第 i 个累积器的漂移率 | 当前模型不估计 LBA 那样的跨试次漂移标准差 |
| `s` | 所有累积器共享的试次内扩散噪声标准差 | 默认 Fixed=1；与 LBA 的跨试次 `sd_v` 不同 |
| `t0` | 非决策时间下端点 | 固定时间或均匀分布下端点 |
| `st0` | 非决策时间总宽度 | `U(t0,t0+st0)`；平均为 `t0+st0/2` |

当前 PsySummary 对漂移要求非负，默认 Lower=0.001。不要将此推广成所有可能的 Wald 竞赛模型在数学上都不允许负漂移；参数域应以具体模型／实现为准。

原始模型参考：[Tillman、Van Zandt & Logan（2020）](https://doi.org/10.3758/s13423-020-01719-6)。rtdists 和 EMC2 都将相应模型描述为独立 Wald 累积器竞赛 [R2–R3]。

### 12.2 软件初始化及边界总表

| 软件／接口 | 初始化方法 | Lower／Upper 的来源 | 对齐限制 |
|---|---|---|---|
| rtdists RDM 帮助页示例 | `A~U(0,1)`，`b=A+U(0,1)`，`t0~U(0,minRT)`，每个 `v~U(0,1)`；`s=1` | 只设置零 Lower，没有有限 Upper；建议不同随机初值重复拟合 | 示例没有估计 `st0`；不能直接归纳出 st0 的“官方随机初值” [R1] |
| EMC2 RDM | 共享贝叶斯链初始化框架；可按用户设定多元正态生成候选点 | 所列参数采用 log 变换；`b=A+B`；自然尺度 Upper 无有限值，实际区域由先验／配置控制 | 文档参数表无 `st0`；省略参数默认常数不是链初值 [R3、S14] |
| HSSM `racing_diffusion_3` | HSSM／PyMC 初始化与用户 initvals；本文不提供未经核实的每参数固定初值 | 当前源码：`A>=0`、`b>=0.1`、各 `v>=0.001`、`t>=0`；Upper 均无限；另查 `b>A` | 已核实内置版本为三响应，有 `t`，没有自由 `s/st0` 参数；默认 HalfNormal 的 sigma 是先验尺度，不是 Upper [R4、L7] |
| 自定义 Stan／其他竞赛框架 | 由具体模型代码及用户初始化控制 | 参数约束和先验决定，没有跨实现统一的 RDM Upper | 只能讨论已给出实现的模型，不能把平台默认无约束初值当作自然尺度默认界 [S16] |

经典 DMC／ggdmc 的 LBA 支持不能自动证明同一版本包含本文的 RDM。fast-dm、DMAT、经典 HDDM 等的 DDM 参数表也不适用于这个模型；本次没有为它们列出未经核实的 RDM 初始化规则。

### 12.3 rtdists 的 `st0` 文档／源码差异

整理时当前 RDM 手册写：非决策时间变异 `st0` 仅在 `rRDM` 随机生成中可用。但同一公开仓库的 `dRDM/pRDM/qRDM` 源码均接收 `st0`，并将其传给 `distribution='wald'` 的 LBA 路径；共享 LBA race 源码中存在对非零 `st0` 的积分分支 [R2、R5、R6]。

因此不能只凭文档宣称“所有当前 rtdists RDM 密度都不支持 st0”，也不能只凭包装器转发就声称已验证该路径在某个发布版本数值正确。本次没有运行 R 的数值验证。本文将其记录为文档与源码不一致、需要固定版本验证的项目。

能明确确认的是：已核实的官方 RDM 拟合示例没有自由 `st0` 初始化。PsySummary 的 `st0` 自动起点属于自己的扩展；该事实不受上述差异影响。

### 12.4 PsySummary RDM 的逐参数规则

| 参数 | 默认模式 | 基础／回退 Value | Lower | Upper | Free 且自动管理时的参考随机范围 |
|---|---|---|---|---|---|
| `A` | Free | 0.5 | 0 | 5 | `U(0,1)` |
| `b` | Free | 1.0 | 0.01 | 10 | `U(A,A+1)` |
| `t0` | Free | `min(0.1,minRT/2)` | 0 | 0.8；与 st0 都 Free 时自动默认 0.5 | `U(0,minRT)` |
| `st0` | Fixed | 0 | 0 | 0.6 | `U(0,min(0.5,minRT))` |
| `s` | Fixed | 1.0 | 0.001 | 10 | 不另设自动参考抽样分布；识别要求 Fixed |
| `v[i]` | Free | `2.0-0.2*(i-1)` | 0.001 | 10 | `U(0,1)` |

参考随机范围与用户边界取交集；没有交集时回退到用户范围。`b` 的生成依赖当前 `A`，随后仍检查 `b>A`，不是分别无约束抽两个值。平均时间约束、手动值保护和多起点规则与第 8 节相同。

这套方案与 rtdists RDM 示例在 `A/b/t0/v` 的基础随机规则上一致，但具有自己的用户边界、种子管理、过滤后逐组处理和 `st0` 扩展，不声称完全复现其优化过程。

### 12.5 RDM 时间支持与尺度识别

`t0/st0>=0`、`b>A>=0`、`s>0`，并遵守当前漂移参数域。连续分布时间支持要求 `t0<minRT`；PsySummary 使用 7 点非决策积分，初始点额外满足 `t0+c7*st0<minRT`。

固定 `s=1` 用于识别证据尺度。自由 `A/b/v` 的数值与 `s` 所选尺度有关，不能直接照搬另一个不同噪声尺度的软件初值或边界。

增大 `st0` 不会使时间分布下端点变小，也不会补救过大的 `t0`。默认 `t0 Upper=0.5、st0 Upper=0.6` 不是要求每组平均非决策时间必须为 0.8 秒，也不是允许所有组都搜索到这两个上界的组合；实际观测时间支持仍然有效。

## 13. 三模型的当前设计汇总

| 项目 | Ratcliff DDM | LBA | RDM |
|---|---|---|---|
| 初始值参考 | rtdists RR98 vignette | rtdists LBA 帮助页拟合示例 | rtdists RDM 帮助页拟合示例 |
| 自动 `t0` 抽样 | `U(0,min(0.5,minRT))` | `U(0,minRT)` | `U(0,minRT)` |
| 自动 `st0` 抽样，若 Free | `U(0,0.5)` | `U(0,min(0.5,minRT))`，本地扩展 | `U(0,min(0.5,minRT))`，本地扩展 |
| t0 单独 Free 的默认 Upper | 0.8 s | 0.8 s | 0.8 s |
| t0/st0 都 Free 的默认 Upper | 0.5／0.6 s | 0.5／0.6 s | 0.5／0.6 s |
| 证据起点参数 | `zr`，绝对 `z=a*zr` | `A`，试次起点 `U(0,A)` | `A`，试次起点 `U(0,A)` |
| 默认尺度固定 | `s=1` | 第一个 `sd_v=1` | `s=1` |
| 联合证据约束 | `a*zr±sz/2` 在 `(0,a)` | `b>A` | `b>A` |
| 非决策时间参数化 | 下端点 t0，带响应侧 d 偏移 | 下端点 t0 | 下端点 t0 |
| 时间积分的初始支持检查 | 9 点，考虑响应侧最短 RT | 7 点 | 7 点 |
| 独立初始似然预检查 | 无 | 无 | 无 |

三模型均使用当前 Filters 后各拟合组的数据，保留手动 Upper 和 Fixed 值；第一候选点保护手动 Free 起点，后续多起点可以随机化自由参数。50 ms 警告不自动修改数据。官方随机范围、产品默认 Upper、数据支持约束应在实现说明中分开陈述。

## 14. LBA／RDM 新增来源及验证边界

| 编号 | 来源 | 支持内容 |
|---|---|---|
| L1 | [rtdists：examples.lba.R](https://raw.githubusercontent.com/rtdists/rtdists/master/examples/examples.lba.R) | 各参数 U(0,1)、b=A+随机间隙、t0 按 minRT 抽样、固定首个 sd |
| L2 | [rtdists README](https://github.com/rtdists/rtdists/) | 另一套 LBA 示例随机范围，未传有限 Upper |
| L3 | [rtdists LBA 手册](https://raw.githubusercontent.com/rtdists/rtdists/master/man/LBA.Rd) | 参数域／接口、t0 下端点、st0 时间变异 |
| L4 | [EMC2：LBA](https://ampl-psych.github.io/EMC2/reference/LBA.html) | 阈值间隙 B、变换、遗漏参数常数和尺度识别 |
| L5 | [ggdmc 作者的三累积器 LBA 教程](https://yxlin.github.io/cognitive-model/lba3/) | LBA 支持、先验、固定 sd_v/st0 的研究示例 |
| L6 | [HSSM 内置模型列表](https://lnccbrown.github.io/HSSM/reference/models-and-likelihoods/) | lba2/lba3/lba4 与 racing_diffusion_3 参数列表 |
| L7 | [HSSM analytical.py](https://raw.githubusercontent.com/lnccbrown/HSSM/main/src/hssm/likelihoods/analytical.py) | LBA／RDM 的 Lower/Upper、参数列表和联合约束 |
| L8 | [StanCon LBA 示例](https://github.com/stan-dev/stancon_talks/blob/master/2018-helsinki/Contributed-Talks/nicenboim/LBA_stancon2018.Rmd) | 自定义 Stan LBA 实现存在；不作为通用参数上界推荐 |
| R1 | [rtdists：examples.rdm.R](https://raw.githubusercontent.com/rtdists/rtdists/master/examples/examples.rdm.R) | 随机起点及 minRT 规则、未传有限 Upper |
| R2 | [rtdists RDM 手册](https://raw.githubusercontent.com/rtdists/rtdists/master/man/RDM.Rd) | 参数定义、s 固定尺度、st0 文档声明 |
| R3 | [EMC2：RDM](https://ampl-psych.github.io/EMC2/reference/RDM.html) | log 变换、自然尺度、B 间隙及省略参数默认值 |
| R4 | [HSSM RDM3 配置](https://raw.githubusercontent.com/lnccbrown/HSSM/main/src/hssm/modelconfig/racing_diffusion_3_config.py) | 内置参数、默认先验和后端 |
| R5 | [rtdists：R/rdm.R](https://raw.githubusercontent.com/rtdists/rtdists/master/R/rdm.R) | d/p/q/rRDM 对 st0 的转发及文档差异 |
| R6 | [rtdists：R/lba_race.R](https://raw.githubusercontent.com/rtdists/rtdists/master/R/lba_race.R) | 共享非零 st0 积分分支，需要固定版本数值验证 |

本次为资料与本地实现核对，未安装／运行上述外部拟合软件，也未重做数值等价性基准。仍待补充的重点为：HSSM 各 race 模型的逐版本实际链初值和固定噪声约定；EMC2 无用户 init_chains 配置时的完整初始化；rtdists RDM 非零 st0 在具体发布版本的数值支持；更多自定义 LBA／RDM 实现的明确参数界。本文仅更新说明文档，没有修改三个模型的拟合代码。
