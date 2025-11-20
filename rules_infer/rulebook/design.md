
VLM对事件的分析结果格式：

```
{{
      { "direct_observation": "...", "causal_inference": "...", "social_event": "intersection_yield", 
      "event_span": {"start":"t-2.0s","end":"t+3.0s"}, "actors": [{"track_id":123, "type":"vehicle", "role":"yielding"}, 
      {"track_id":45, "type":"pedestrian", "role":"crossing"}], 
      "context": {"junction":"signalized", "tl_state":"red", "crosswalk":true}, 
      "evidence": ["pedestrian enters crosswalk", "ego decelerates from 8m/s to 0"], 
      "confidence": 0.82, "alternatives": ["congestion_stop"], "map_refs": {"lane_ids":[...], "crosswalk_id":...} }
}}
```
### social_event按照层级方式划分为两层，包括：
* **1.Right‑of‑Way 交换：** intersection_yield / crosswalk_yield / roundabout_merge_yield / bus_stop_merge_yield / U_turn_yield
* **2.竞争/并入：** merge_compete / zipper_merge
* **3.轨迹相对关系：** cut_in / cut_out / follow_gap_opening
* **4.障碍与绕行：** double_parking_avoidance / blocked_intersection_clear / dooring_avoidance
* **5.行人/非机动车：** jaywalking_response / bike_lane_merge_yield
* **6.紧急与礼让：** emergency_vehicle_yield / courtesy_stop
* **7.拥堵态：** congestion_stop / queue_discharge
* **8.合规/违规：** red_light_stop / stop_sign_yield / priority_violation
* **9.开放类：** novel_event（VLM 需给出 proposed_label+理由）

对上述层级化的social event保留开放类与自增长机制，理由：1.闭集能带来一致的监督、更稳的评测与跨场景对齐； 2.开集回退能吸纳长尾与地区性习惯，不把模型“绑死”在有限类别上。
VLM标注时的策略：让 VLM 优先在闭集里选，若置信 < $\tau$ 或不匹配，则回退到 novel_event，并要求填： proposed_label（自由文本）、nearest_parent（从上面一级节点中选）、rationale（映射理由）。
同时对 social_event 增加先验与一致性检查（与地图、信号状态、角色是否匹配），可过滤幻觉

* **C‑1 事件/关系本体与数据质量层（EventBank & QC）**：维护 `social_event / relation / role` 的层级本体、版本与一致性规则；管控 VLM 输出质量。
* **C‑2 可靠性与运行时安全层（Reliability Layer）**：不确定性校准、实时剪枝/蒸馏、在线安全约束（TTC、越界、碰撞约束）与可视化调试。

---

## M‑1 **ContextWeaver**（环境与时空编码器）

**目标**：把传感器、agent 历史、地图、天气/时段整合为统一的时空表示，供后续风格/图/因果/解码器共享。

### 输入与表征

* **Agent 历史**（每体 T 帧）：位置/速度/加速度/航向 + 可见性/遮挡置信；
    
    **A) 变道/横向机动倾向（Lane‑Change Propensity, LCP）**

    * **变道尝试率**：在时间窗 $[t-\Delta,t]$，若

      * 与最近车道中心线的横向偏移 $d_\perp$ 在 $\epsilon_d$ 内反复靠近/远离（例如零交叉>2 次），且
      * 横向速度 $v_y$ 超过阈值 $\epsilon_v$ 持续 > $\tau$，
        则记一次“尝试”；LCP = 尝试次数 / 时长。
    * **成功率/放弃率**：尝试后 $\le 3$s 是否完成车道 ID 切换；否则记“放弃”。

    **B) 并线“可接受间隙”（Gap Acceptance）**

    * 在并入或换道目标车道上，定义潜在“前/后随车” $f,b$；
      * 计算到二者的**等效时间间隙**（沿目标车道方向）：
        $\mathrm{TTA}_f = \frac{\text{沿道距离}(A\to f)}{\max(\epsilon, v_f)}$、
        $\mathrm{TTA}_b = \frac{\text{沿道距离}(b\to A)}{\max(\epsilon, v_A)}$；
      * 当 agent 实际完成并入时，记录被**接受**的 $\min(\mathrm{TTA}_f,\mathrm{TTA}_b)$；当出现明显“探测—放弃”（接近边界→回退）时，记录**拒绝**的间隙；
      * 拟合个体的“间隙阈值分布” $G_i(\cdot)$（如对数正态），其位置参数可作为**个体保守/激进**量化。

    **C) 跟驰习惯（Car‑Following Style）**
    * 头距/时间头距：$\mathrm{THW} = \frac{\text{距离}(A\to f)}{\max(\epsilon, v_A)}$，统计均值/分位数；
      * 危险接近频率：$\mathrm{TTC}<\tau$ 的占比；
      * 加速度响应增益：当前向速度误差 $(v_f-v_A)$ 与纵向加速度 $a_x$ 的回归斜率（类似 IDM/OVM 的经验拟合）。
  
    **D) 合规/礼让度（Compliance & Courtesy）**
    * **红灯/停止线合规**：当 $\text{tl}=\text{red}$ 且距离停止线 $d<d_0$ 时，若 $\min_t v_t < v_{\min}$ 且停车点在停止线前 $\le \delta$ 范围，记一次合规；统计合规率。
      * **斑马线礼让**（有行人进入/在内）：在 $[t_\text{enter}-1s, t_\text{stop}+1s]$ 计算**必要减速度** $a_\text{req}$（保证行人与车最小安全距离/时间阈值），与实际减速度 $a_\text{obs}$ 的差：
        $\text{Courtesy Margin}=\max(0, a_\text{obs}-a_\text{req})$；统计其分布/均值。
      * **速度合规**：相对限速偏差 $\Delta v = v - v_\text{limit}$ 与越界占比。

    **E) 攻击性/平顺性综合指数（Aggressiveness/Comfort Index）**
  * 归一化后线性或非线性汇总：

    $$
    \text{AggIndex} = w_1 \cdot \text{jerk95} + w_2 \cdot \text{minTTC}^{-1} + w_3 \cdot \text{GapThr}^{-1} + w_4 \cdot \text{speeding\_rate}
    $$

    权重可从数据学习（例如用人类主观标注/聚类作为弱监督）。
  以上 A–E 的统计序列（加上基础运动学）就是个体风格编码器的输入；它们也能直接作为解释指标对外可视化。
* **BEV 感知**（可选）：多帧 LiDAR/RGB‑to‑BEV 栅格（静态+动态通道）；
  * **向量化地图**：lanelet 片段/中心线、车道边界、停止线、斑马线、交叉口多边形、限速/优先级；
  * **天气/时段/地域**：离散 embedding（one‑hot/learned）；
  * **时间编码**：相对时间 $t{-}t_0$、信号相位（如已知）。

### 结构与融合

* **Agent 编码器**：TCN/Transformer（维度 d=128–256），输出 per‑agent token `h_i^0`；
* **Map 编码器**：对 lanelet/区域多段线做 Polyline‑Transformer；每个 Lane/Zone 输出 token `h_L, h_Z`；
* **BEV 编码器**（可选）：小型 CNN→Patch tokens，与向量化地图 Cross‑Attention；
* **跨模态融合**：

  * **Agent↔Lane/Zone** 双向 Cross‑Attention（半径/拓扑裁剪后的邻域）；
  * **天气/时段/地域** embedding 作为 FiLM/条件偏置注入到注意力与 LayerNorm。
* **输出**：时间维的 `H_A(t)`, `H_L(t)`, `H_Z(t)`（供风格/图/解码器使用）。

### 预训练任务（强烈建议先做）

* 掩码轨迹重建、未来朝向预测、lane 关联分类、BEV 静态/动态分割自监督、地图—感知配准一致性。

---

## M‑2 **StyleLattice**（个体/群体风格学习层）

**目标**：学习可解释、可控的层次风格：**个体** $z_i$ 与**群体** $z_g$。

### 输入信号

* **个体风格特征**（由 ContextWeaver得到）：
* **群体风格特征**：按 Zone/Lane 聚合同窗内多体的上述统计 + 流量/密度指标。
* **上下文**：`c={zone_type, region, time_of_day, weather, lane_type}`。

### 模型结构

* **IG‑VAE(+VQ) 组合**：

  * `E_i`（个体编码器）：时序 Transformer + MLP → 输出 `μ_i, σ_i`（连续 $z_i$）与离散 code `q_i`（VQ，K=32–128）；
  * `E_g`（群体编码器）：Set Transformer/DeepSets 聚合 Zone 内统计 `s_Z` + 上下文 `c` → `μ_g, σ_g` 与 `q_g`（VQ）；
  * **条件先验**：`p(z_g|c)=N(μ_c,Σ_c)`，`p(z_i|actor,visibility)`。
* **正则与对齐**：β‑VAE（β≈2–4）+ VQ commit + InfoNCE（同 c 的不同 Zone 拉近 $z_g$）+ **统计匹配**（用 MMD/KL 让生成行为在该 $z_g$ 条件下复现“平均礼让时距/加塞频率”等）。
* **记忆库**：按 `c` 维护 EMA 的群体原型，供冷启动与部署端快速适配。
* **对齐训练**：
  * 跨场景对比：相同$c$的不同zone拉近$z_g$，不同$c$拉远
  * 统计匹配：用MMD/KL让模型在该$z_g$条件下重建聚合统计（如平均礼让时距离、加塞频率）
* **时间平滑与记忆**：给每个$c$维护指数滑动平均的$z_g$(或记忆库key=$c$)，加稳定正则
**输出**：每体 `z_i`（连续+离散）与当前 Zone 的 `z_g`，不会直接解码到动作，而是作为**条件**提供给 M‑3/4/5。

---

## M‑3 **SocioTopo‑ALZ**（Agent–Lane–Zone 三部图）

**目标**：在“谁–围绕哪条车道/哪个冲突区–以何种关系互动”的图上，进行消息传递与关系推断。

### 图构建

* **节点**：`A`（agents）、`L`（lanelet 段）、`Z`（冲突/功能区：交叉口冲突区、并线区、斑马线、公交港湾、临时障碍区等）。
* **边**（候选 + 时变）：

  * `A–L`：on/adjacent/targetable/near\_stop\_line/near\_crosswalk（多由几何确定）；
  * `A–Z`：will\_enter / yielding\_to / competing\_for / violating / unaffected（潜在，需学习）；
  * `A–A`：follow / yield\_to / cut\_in / cross\_path / leader\_of\_platoon / none（潜在，需学习）；
  * `L–Z`：enters\_conflict / priority / protected\_turn / stop\_controlled（由拓扑/规则确定）。
* **边特征**：$相对纵横向位移(Δs,Δd)$、曲率、剩余距离至 stop_line、TTZ(Time to Zone)/TTC、相对速度、遮挡、优先级/限速、信号相位、Zone 类型。

### 关系学习

* **确定边**：几何/地图能确定的边直接固定；
* **潜在边**：对 `A–A`、`A–Z` 训练 $\pi_e=softmax(f_θ(x_e))$，用 **Gumbel‑Softmax** 采样可微、加**稀疏正则**与**先验约束**（如 `yield_to` 反对称、`follow` 近似传递）。
* **弱监督来源**：VLM 的 `social_event/actors/roles/event_span` 作为片段级标签；时序一致性正则。
* **自监督对比**：正样：同事件窗口内边类型一致；负样：跨场景/跨城市错配

这个三部图阐述“谁与谁、围绕哪条车道/哪个冲突区、以什么社会关系互动”的思想

#### 用图注意力建模“风险→意图”的因果关系
**风险特征**：`TTC/TTZ`、相对速度、冲突区占用预测、可见性遮挡、优先级。
在 R‑GAT/HGT 的注意力打分中加入**可解释的偏置项**：

$\alpha_{u\to v}^{(r)}=\mathrm{softmax}_u\Big(\frac{Q_r h_u\cdot K_r h_v}{\sqrt{d}}+ \beta_1 \cdot \phi_{\text{risk}}(u,v)+ \beta_2 \cdot \phi_{\text{priority}}(u,v)- \beta_3 \cdot \phi_{\text{implausible}}(u,v)\Big)$

* $φ_{risk}$: 递减函数，如 `exp(-TTZ/τ)` 或 `1/(1+TTC)`；
* $φ_{priority}$: 基于路权（主路>支路>行人先行等）；
* $φ_{implausible}$: 物理不可能/遮挡严重的边惩罚。
  训练时对 `β` 加 L1 正则并在解释界面输出 `β·φ` 的贡献，直观呈现“风险”如何推动意图变化。

### 图网络

* **Heterogeneous Graph Transformer (HGT) / R‑GAT**：
  关系特定投影 + 注意力；注意分数加**风险/优先级偏置**（见 M‑4）；
* **Zone 超边聚合**：Zone 往往联结多 agent，我们引入超边或二级聚合：
  * 先把所有连接到同一 Z_k 的 A 节点做一次冲突级池化（max/attention），形成 $h_{Z_k}^{A}$
  * 再把 $h_{Z_k}^{A}$ 与 $Z_k$ 自身语义 $h_{Z_k}$ 融合，回写给各个参与的 agent
* **输出**：更新后的 $H_A,H_L,H_Z$，以及潜在边类型分布 $\pi_e$。$m_{i}^{A↔A}, m_{i}^{A↔L}, m_{i}^{A↔Z}$ 三路消息 + $z_i, z_g$ + 自身编码 $h_i$ 拼接，作为后续意图/轨迹/事件的条件向量。
---

## M‑4 **InterveneNet**（因果结构学习与反事实推理）

**目标**：把“风险→意图/轨迹”的因果结构显式化，并支持**可微反事实**。

### 结构方程（概念）

$$
\begin{aligned}
\text{intent}_i &\leftarrow f\big(h_i, z_i, z_g, m_i^{A↔A}, m_i^{A↔L}, m_i^{A↔Z}\big)\\
\text{edge_type}_{e} &\leftarrow g(x_e, z_g)\\
\end{aligned}
$$

其中 $m_i^{*}$ 是 SocioTopo‑ALZ 的消息聚合。

### 因果注意力与可微干预

* **注意力偏置**：在 HGT/R‑GAT 的 logits 中加入
  `+ β₁·φ_risk(TTZ/TTC) + β₂·φ_priority(路权) − β₃·φ_implausible(遮挡/物理不可能)`，并学习 `β`；
* **门控因素**：为候选原因 $c\in\{$行人存在、信号灯状态、可见性、间隙大小、邻车速度等$\}$ 设门控 $\alpha_c\in[0,1]$，作用到相关特征/消息；
* **干预实现**：`do(c=c')` = 改输入或置 $\alpha_c$ 为 0/替换，再前向得到新意图/轨迹；
* **方向性/忠实损失**：

  * 反事实方向：若 VLM `causal_inference` 指“因行人进入而停车让行”，则 `remove pedestrian` 后 `stop/yield` 概率应下降（排名/hinge）；
  * 证据对齐：VLM `evidence` 的时空区域应与注意/显著性覆盖。
* 
* **可微反事实：** 对 L/Z 节点或 A 节点进行软干预（例如改交通灯状态、移除/晚到一个行人、缩短跟驰间隙），比较预测意图/轨迹/规划的变化，形成因果解释分数。
* **冲突风险联动：** 以冲突区为桥，使用图注意/GAT 进行风险与意图的因果建模 

### 意图表示

意图用可组合的“关系-目标“图（Compositional Intent Graph, CIG）来表示，把意图表述为三元组集合，包含{relation,target,goal}，
其中：
* **relation** $\in$ {yield_to,take_gap_from,follow,block,...}(来自A-A/A-Z边类型)
* **target** $\in$ {特定 agent/Zone/Lane，或集合（例如“进入同一 Zone 的队列头”）}
* **goal** $\in$ {reach(lane_id), keep_distance(d), maintain_speed(v),...}带连续参数
* **执行条件/终止条件**：如 until enter(z), until gap>$\tau$
* **学习方式：**
  * 从三部图读出relation(分类)
  * target由指针网络在候选集合中指派
  * goal的参数做回归/分布预测
  * 用弱监督（VLM 事件/角色/时段）+ 行为一致损失（生成的轨迹应满足这些单元规则，见下）

* **优势：** 可组合、可解释、能描述“又礼让又并线择机”的复合意图。
* **一致损失(满足规则)**
  * 对每个单元，构造可微约束罚项（如 yield_to(j) ⇒ 规划/预测中 $THW_j\geq \tau$且 ego 优先级劣势下纵向速度不超越）；
  * 用 soft‑logic（Gödel/Łukasiewicz t‑norm）将多约束组成 differentiable 目标，兼容反事实验证。
---


## M‑5 **PolyDeco**（解码器族）

### 5.1 **TrajDiff**（多智能体多模态轨迹扩散解码器）

* **条件**：[$H_A ⊕ z_i ⊕ z_g ⊕ π_e ⊕ intent/CIG$]；
* **生成**：DDPM/Flow‑Matching（点序列回归），每个agent输出 `M` 条模式（M=6–12）及协方差；
* **一致性**：扩散步内做 1–2 轮跨体消息传递，软碰撞/越界惩罚；
* **可控性**：固定 $z_g$/`intent` 可生成指定风格/意图下的轨迹组。

### 5.2 **EventSpanter**（社会事件/时段/角色解码器）

* **事件/角色**：序列标注 + 指派（Hungarian）或多标签分类；
* **时段**：指针回归 `start/end`（IoU 损失）；
* **一致性**：与 `π_e`、intent/CIG 的逻辑一致约束。

### 5.3 **CIG‑Composer**（可组合意图图头）

* **表示**：若干三元组 `(relation, target, goal)`（目标可为 agent/Zone/Lane，goal 含连续参数与终止条件）；
* **实现**：关系分类 + 指针网络选 target + MLP 回归 goal 参数；
* **规则满足**：Soft‑logic 罚项确保预测轨迹/规划满足这些单元（如 `yield_to(j)` ⇒ 与 j 的 THW≥τ）。

### 5.4 **PlanCrafter**（Ego 规划器）

* **方式**：IL+代价学习（或 MPC 外环），代价 = 安全/舒适/进度 + **风格/规范项**（由 $z_g$ 参数化），并接受 CIG 作为软约束；
* **Human‑like 目标**：

  * **Fréchet Behavior Distance (FBD)**：Ego 规划在行为编码器空间与人类分布对齐；
  * **Norm Compliance**：礼让阈值/停车距离/合流间隙偏差惩罚；
  * **反事实理性**：干预后规划变化与因果方向一致。

---

## M‑6 **OmniLoss**（损失函数与训练日程）

### 主要损失

$ L = \lambda_{\text{traj}} L_{\text{NLL/EMD}}+ \lambda_{\text{event}}(L_{\text{event‑CE}}+L_{\text{span‑IoU}}+L_{\text{role}}) \\ + \lambda_{\text{edge}}(L_{\text{edge‑CE}}+\text{sparsity}+\text{prior‑consistency}) \\$
$+ \lambda_{\text{cf}}(L_{\text{direction}}+L_{\text{evidence‑align}}) \\ + \lambda_{\text{style}}(L_{\text{KL}}+L_{\text{VQ‑commit}}+L_{\text{InfoNCE}}+L_{\text{stats‑match}}) \\ + \lambda_{\text{plan}}(L_{\text{FBD}}+L_{\text{norm}}+L_{\text{safety}})$

### 训练流程（推荐）

1. 预训练 **ContextWeaver**（自监督）。
2. 训练 **SocioTopo‑ALZ** + 潜在边类型（引入 VLM 弱标签）。
3. 加入 **StyleLattice**（IG‑VAE+VQ）并与图联合训练。
4. 训练 **EventSpanter / CIG‑Composer / InterveneNet**（反事实损失开启）。
5. 接入 **TrajDiff** 多模态生成；
6. 最后端到端微调并加入 **PlanCrafter** 的规划损失。

---

## C‑1 **EventBank & QC**

* **内容**：`social_event` 层级本体（含开放类 `novel_event`）、边/角色/终止条件模板、参数范围；
* **作用**：

  * 为 VLM 提示提供**固定枚举 + 开放回退**；
  * 规则检查（与地图/信号对齐、角色一致、时间逻辑）；
  * 版本管理与评测对齐（不同版本本体的映射）。

## C‑2 **Reliability Layer**

* **不确定性校准**：温度缩放/Dirichlet 校准到意图/事件/碰撞概率；
* **实时性**：邻域裁剪、Zone 稀疏化、图蒸馏（把 HGT 蒸到轻量 MHA）；
* **安全外环**：基于 TTC/越界/碰撞的 hard‑constraint 过滤规划；
* **可视化与审计**：显示 `evidence` ↔ 注意/CIG 单元满足度 ↔ 反事实效果。

---

# 每个关键部分的**实现要点一览**

### ContextWeaver（实现要点）

* 邻域裁剪：Agent↔Lane/Zone 半径 30–60m；帧窗 T=6–8；
* Token 维度 d=192；8 头注意；
* 地图向量化：polyline 分段 ≤ 20 个点；
* 输出缓存：`H_A/H_L/H_Z` 供后续模块。

### StyleLattice（实现要点）

* 个体统计以 1s/0.5s 滑窗计算，Savgol 平滑；
* IG‑VAE 潜变量维度：$z_i\in \mathbb{R}^{16}$、VQ K=64；$z_g\in \mathbb{R}^{16}$、VQ K=32；
* 统计匹配：对“礼让裕量/可接受间隙/THW”的分布做 MMD（RBF 核）。

### SocioTopo‑ALZ（实现要点）

* Zone 来源：**地图优先**（交叉/合流/斑马线/环岛/港湾），BEV 检测补临时 Zone；
* 潜在边学习：Gumbel‑Softmax 温度从 2→0.5 退火；
* 约束：`yield_to` 反对称损失，`follow` 的链一致性正则。

### InterveneNet（实现要点）

* 注意偏置函数：`φ_risk=exp(-TTZ/τ)` 或 `1/(1+TTC)`；
* 反事实采样：每个片段随机 1–2 个因素做 `do(·)`；
* 方向性损失：`max(0, margin - (score_base - score_cf))`。

### PolyDeco（实现要点）

* TrajDiff：UNet1D（点序列）、每体 M=8 模式；Social‑NCE 促使模式差异化；
* EventSpanter：边界指针 + 事件 CE；
* CIG‑Composer：

  * 关系头（softmax 10–20 类）、
  * 指针网络在邻居/Zone/Lane 候选集上选 target、
  * goal 参数（如期望间隙/速度/停止距离）用高斯回归；
  * Soft‑logic 将 `yield_to/merge_take_gap` 等规则转成可微满足度（训练与解释共用）。
* PlanCrafter：

  * 行为编码器 `B(·)`：小 Transformer，输出 32 维；
  * FBD 在 `B(·)` 空间与人类分布对齐；
  * Norm‑Compliance 用 $z_g$ 提供门限（礼让阈值、合流间隙、停止距离等）。

### OmniLoss（实现要点）

* 权重初值：`λ_traj=1, λ_event=1, λ_edge=0.5, λ_cf=0.5, λ_style=0.5, λ_plan=0.5`；
* 分阶段训练再端到端微调，学习率从 1e‑3→1e‑4 余弦退火；
* 混合精度 + 梯度累积，图邻域上限（每体邻居≤12）。

---

