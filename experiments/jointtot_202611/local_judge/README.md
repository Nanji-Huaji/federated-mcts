# 本地评估器大小对照（新推理）

这不是又一次翻旧标签：从已有 Qwen3.7-plus 首层候选中取 20 道固定验证题、每题四个合法且状态不同的候选，**新运行 Qwen2.5-7B-Instruct 和 Qwen2.5-32B-Instruct**。每一题按预先固定的顺序、及其逆序各做一次。两个裁判看到的文字相同，不看原投票或正确性；比较 A/B/C/D 四个标签的下一 token logits，不生成解释、不续跑子树。每个模型 40 条提示，共 80 条，不是 80 道独立试题。候选冻结和评价指标见[协议](PROTOCOL.md)。

**结果。** 原顺序首选命中：7B 为 8/20，32B 为 15/20（仅 32B 对 8 题、仅 7B 对 1 题；按题差 +35 点，描述性 95% 区间 [+10,+60]）。反序后两者同为 12/20，首选仍是同一个状态的题数分别只有 1/20、12/20。这个结果支持检查位置偏差，不能单凭一次呈现主张强评估器处处更优。有解可选的池为 18/20，均匀随机首选期望 9.5/20。

- [输入](artifacts/inputs.json)、[7B 原分数](artifacts/scores_7B.json)、[32B 原分数](artifacts/scores_32B.json)；每份分数含模型快照、权重分片标识、提示哈希、tokens 和版本。
- [逐题选择及后验算术标签](artifacts/summary.json)、[配对差和独立检查](artifacts/checks.json)。主指标为 20 题中首选仍能在两步内到 24 的题数；原池根本没有活路的题也保留。次指标为前二至少含活路，以及反转选项顺序后选中同一状态的次数。不能用它代表标准有理数 Game24、整套 ToT 求解率，或宣称云 API 已测。
- 模型用本机 CPU 的 bfloat16 前向计算，未联网、未量化、未改仓库环境。耗时写入原分数方便安排复跑，不能当部署时延结果。两模型 token 级前向虽然相同，但不是等算力比较。

只复核输入、分数、标签和区间（无需模型）：

~~~bash
D=experiments/jointtot_202611/local_judge
python3 -B "$D/verify.py" --check
~~~

若要重新推理，先选一个**新的**输出目录，别覆盖随附分数。需要本地已经缓存两个协议指定的 Hugging Face 快照，PyTorch 2.7.0、Transformers 4.53.2 是本次用的版本。建议 70+ GiB 可用内存；两模型顺序执行。这里的 PY 必须指向装有该环境的 Python。

~~~bash
D=experiments/jointtot_202611/local_judge
PY="$HOME/miniconda3/envs/vllm/bin/python"  # 或自己的同版本环境
OUT=$(mktemp -d)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python3 -B "$D/run.py" prepare --out "$OUT"
for model in 7B 32B; do "$PY" -B "$D/run.py" infer --size "$model" --out "$OUT" --threads 12 --batch 4; done
python3 -B "$D/run.py" summary --out "$OUT"
python3 -B "$D/verify.py" --out "$OUT"
~~~

同一题的两个呈现不是独立样本；按 20 对题计算差值，在经验三项分布上精确枚举配对 bootstrap 分位数。后验标签还用独立子集 DP 对每个状态交叉核验。跨后端的低精度并列/排序可变，以随附原始 logits 为本次观测。
