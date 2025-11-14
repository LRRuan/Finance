# Finance 项目总览

本仓库聚合了完整的金融智能体工作流：  
- `Financial-MCP-Agent/`：LangGraph 驱动的多智能体分析客户端  
- `a-share-mcp-is-just-i-need/`：通过 MCP 协议暴露的 A 股数据服务  
- `Fin-R1* / qwen_* / nasdaq_*`：需要手动下载的模型与数据资源

> ⚠️ GitHub 版本仅包含核心源码（主要是 `src/main.py` 与 `mcp_server.py`），**训练数据与模型权重均需自行下载并放置到对应目录**。

---

## 目录结构

```
Finance/
├── Financial-MCP-Agent/        # 多智能体客户端
├── a-share-mcp-is-just-i-need/ # MCP 数据服务
├── Fin-R1/                     # 手动下载：全精度权重
├── Fin-R1-awq/                 # 手动下载：AWQ 量化权重
├── Qwen3-1.7b/                 # 手动下载：情感与风险模型base
├── qwen_sentiment_model/       # 脚本训练
├── qwen_risk_model/            # 脚本训练
├── nasdaq_news_sentiment/      # 手动下载：新闻情感 CSV
├── risk_nasdaq/                # 手动下载：风险 CSV
└── requirements.txt            # Python 依赖
...训练脚本
```

---

## 运行前准备

1. **Python/Conda**
   - 推荐使用 `conda create -n Finance python=3.10`
   - 依赖：`pip install -r requirements.txt`
2. **环境变量**
   - 在 `Financial-MCP-Agent/.env` 中设置  
     ```
     OPENAI_COMPATIBLE_BASE_URL=
     OPENAI_COMPATIBLE_MODEL=
     OPENAI_COMPATIBLE_API_KEY=
     USE_LOCAL_MODEL=vllm    # local / vllm / api
     ```
3. **模型与数据（需手动下载）**
   - Fin-R1 全量与 `Fin-R1-awq`（`git lfs clone` 自 HuggingFace）
   - `qwen_sentiment_model/` 与 `qwen_risk_model/`：放置 LoRA/Adapter
   - `nasdaq_news_sentiment/sentiment_deepseek_new_cleaned_nasdaq_news_full.csv`
   - `risk_nasdaq/risk_deepseek_cleaned_nasdaq_news_full.csv`

---

## 启动顺序与关键命令

1. **启动 MCP 数据服务**（在 `a-share-mcp-is-just-i-need/` 目录）
   ```
   uv run python mcp_server.py
   ```
   > 该服务会通过 http 启动所有 A 股数据工具，供 LangGraph Agent 调用。

2. **启动 Fin-R1 vLLM 推理服务**（在 `Finance/` 根目录）
   ```
   vllm serve ./Fin-R1-awq \
       --host 0.0.0.0 \
      --port 8000 \
      --max-model-len 16500
   ```
   > 如需使用全精度 Fin-R1，可将路径替换为 `./Fin-R1` 并酌情调整显存参数。

3. **运行多智能体主流程**（在 `Financial-MCP-Agent/` 中）
   ```
   $(CONDA_PATH) run -n Finance --no-capture-output python -m src.main --command "贵州茅台值得投资吗"
   ```
   > `$(CONDA_PATH)` 为 `conda` 可执行文件路径，可通过 `which conda` 获取。指令也可替换为其他自然语言问题。

运行完成后：
- 报告输出至 `Financial-MCP-Agent/reports/`
- 执行日志输出至 `Financial-MCP-Agent/logs/`

---
