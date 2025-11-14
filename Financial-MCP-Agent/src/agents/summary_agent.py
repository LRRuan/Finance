"""
Summary Agent: Consolidates analyses from other agents into a final report.
汇总 Agent：将其他 Agent的分析结果整合成最终报告
"""
import os
import time
from typing import Dict, Any
from langchain_openai import ChatOpenAI  # 恢复OpenAI导入
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re

from src.utils.state_definition import AgentState
from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from dotenv import load_dotenv

# 从.env文件加载环境变量
load_dotenv(override=True)

logger = setup_logger(__name__)

from transformers import StoppingCriteria, StoppingCriteriaList
import time
import logging

logger = logging.getLogger(__name__)


class ProgressStoppingCriteria(StoppingCriteria):
    """
    一个自定义的StoppingCriteria，用于在generate过程中打印进度。
    """

    def __init__(self, total_tokens: int, print_step: int = 10):
        """
        Args:
            total_tokens (int): 预期的总生成token数 (max_new_tokens)。
            print_step (int): 每生成多少个token打印一次进度。
        """
        self.total_tokens = total_tokens
        self.print_step = print_step
        self.generated_tokens = 0
        self.start_time = time.time()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # __call__方法在每生成一个token后被调用
        self.generated_tokens += 1
        if torch.cuda.is_available():
            allocated_memory = torch.cuda.memory_allocated()  # in bytes
            reserved_memory = torch.cuda.memory_reserved()  # in bytes

        # 每隔 print_step 个 token，就打印一次进度
        if self.generated_tokens % self.print_step == 0:
            elapsed_time = time.time() - self.start_time
            avg_speed = self.generated_tokens / elapsed_time
            progress = (self.generated_tokens / self.total_tokens) * 100

            # 使用logger打印，这样可以和你的其他日志保持一致
            logger.info(
                f"推理进度: {self.generated_tokens}/{self.total_tokens} tokens ({progress:.2f}%) | "
                f"速度: {avg_speed:.2f} tokens/sec"
                f"Allocated memory: {allocated_memory / (1024 ** 2):.2f} MB"
                f"Reserved memory: {reserved_memory / (1024 ** 2):.2f} MB"
            )

        # 我们不希望这个标准来停止生成，所以总是返回 False
        return False


def truncate_report_at_baseline_time(report_content: str, current_time_info: str) -> str:
    """
    使用正则表达式截断报告，在"分析基准时间"那一行之后停止
    
    Args:
        report_content: 完整的报告内容
        current_time_info: 当前时间信息
    
    Returns:
        截断后的报告内容
    """
    # 构建多种可能的"分析基准时间"模式
    baseline_patterns = [
        rf'分析基准时间[：:]\s*{re.escape(current_time_info)}',
        rf'分析基准时间[：:]\s*{re.escape(current_time_info)}\s*$',
        rf'基准时间[：:]\s*{re.escape(current_time_info)}',
        rf'时间基准[：:]\s*{re.escape(current_time_info)}',
        rf'分析时间[：:]\s*{re.escape(current_time_info)}',
        rf'报告时间[：:]\s*{re.escape(current_time_info)}',
        rf'生成时间[：:]\s*{re.escape(current_time_info)}',
        rf'更新时间[：:]\s*{re.escape(current_time_info)}',
        rf'数据时间[：:]\s*{re.escape(current_time_info)}',
        rf'分析基准[：:]\s*{re.escape(current_time_info)}'
    ]
    
    # 尝试匹配各种模式
    for pattern in baseline_patterns:
        match = re.search(pattern, report_content, re.MULTILINE | re.IGNORECASE)
        if match:
            # 找到匹配位置，截断到该行的末尾
            end_pos = match.end()
            
            # 查找该行的结束位置（换行符）
            line_end = report_content.find('\n', end_pos)
            if line_end == -1:
                # 如果没有换行符，说明是最后一行，直接截断
                truncated_content = report_content[:end_pos].strip()
            else:
                # 截断到该行结束
                truncated_content = report_content[:line_end].strip()
            
            logger.info(f"截断报告在'分析基准时间'行之后，截断位置: {end_pos}")
            return truncated_content
    
    # 如果没有找到匹配的模式，尝试查找包含时间信息的行
    time_patterns = [
        rf'.*{re.escape(current_time_info)}.*',
        rf'.*{re.escape(current_time_info.split()[0])}.*',  # 只匹配日期部分
        rf'.*{re.escape(current_time_info.split()[1])}.*'   # 只匹配时间部分
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, report_content, re.MULTILINE | re.IGNORECASE)
        if match:
            end_pos = match.end()
            line_end = report_content.find('\n', end_pos)
            if line_end == -1:
                truncated_content = report_content[:end_pos].strip()
            else:
                truncated_content = report_content[:line_end].strip()
            
            logger.info(f"截断报告在时间信息行之后，截断位置: {end_pos}")
            return truncated_content
    
    # 如果都没有找到，返回原始内容
    logger.warning("未找到'分析基准时间'模式，返回原始报告内容")
    return report_content


def load_finr1_model(model_path="/home/ruan/Finance/Fin-R1"):
    """加载FinR1模型"""
    logger.info(f"{WAIT_ICON} Loading FinR1 model from {model_path}...")
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        total_mem_bytes = torch.cuda.get_device_properties(0).total_memory
        # 转换为GB
        total_mem_gb = total_mem_bytes / (1024 ** 3)

        # 留出非常小的buffer，比如0.5GB，剩下的全部给模型
        # 这是一种比较激进的策略，告诉accelerate“大胆用！”
        max_memory_gb = total_mem_gb - 0.5

        logger.info(f"GPU总显存: {total_mem_gb:.2f}GB. 设置最大使用: {max_memory_gb:.2f}GB")
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            # max_memory={0: f'{max_memory_gb:.2f}GB'},  # <--- 关键！明确告诉accelerate GPU 0 能用多少内存
            trust_remote_code=True
        )
        
        model.eval()
        logger.info(f"{SUCCESS_ICON} FinR1 model loaded successfully")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"{ERROR_ICON} Failed to load FinR1 model: {e}")
        raise e


def generate_report_with_finr1(model, tokenizer, prompt, max_new_tokens=5000):
    """使用FinR1模型生成报告"""
    
    try:
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        progress_criteria = ProgressStoppingCriteria(total_tokens=max_new_tokens, print_step=10)
        # 生成预测
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.5,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([progress_criteria])
            )
        
        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的报告部分（移除输入提示）
        # 方法1：尝试通过字符串匹配移除输入提示
        if prompt in generated_text:
            report = generated_text[len(prompt):].strip()
        else:
            # 方法2：如果字符串匹配失败，尝试通过token长度来提取
            input_length = len(tokenizer.encode(prompt, return_tensors="pt")[0])
            output_length = len(outputs[0])
            
            if output_length > input_length:
                # 只保留新生成的部分
                new_tokens = outputs[0][input_length:]
                report = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            else:
                # 如果无法确定，返回完整文本但尝试清理
                report = generated_text.strip()
        
        return report
    
    except Exception as e:
        logger.error(f"{ERROR_ICON} Error generating report with FinR1: {e}")
        raise e


def get_model_choice():
    """获取模型选择，默认选择API"""
    # 可以通过环境变量控制模型选择
    model_choice = os.getenv("USE_LOCAL_MODEL", "api").lower()
    return model_choice


async def summary_agent(state: AgentState) -> Dict[str, Any]:
    """
    整合基本面、技术面和估值分析的结果
    使用LLM生成最终的综合性报告
    """
    logger.info(f"{WAIT_ICON} SummaryAgent: Starting to consolidate analyses.")

    # 获取执行日志记录器，用于记录 Agent的执行过程
    execution_logger = get_execution_logger()
    agent_name = "summary_agent"

    # 从状态中提取当前数据、消息和用户查询
    current_data = state.get("data", {})
    messages = state.get("messages", [])
    user_query = current_data.get("query", "")

    # 记录 Agent开始执行，包含可用的分析类型
    execution_logger.log_agent_start(agent_name, {
        "user_query": user_query,
        "available_analyses": {
            "fundamental": "fundamental_analysis" in current_data,
            "technical": "technical_analysis" in current_data,
            "value": "value_analysis" in current_data,
            "news": "news_analysis" in current_data
        },
        "input_data_keys": list(current_data.keys())
    })

    # 记录 Agent开始时间，用于计算执行时长
    agent_start_time = time.time()

    # 获取之前 Agent的分析结果
    fundamental_analysis = current_data.get(
        "fundamental_analysis", "Not available")
    technical_analysis = current_data.get(
        "technical_analysis", "Not available")
    value_analysis = current_data.get("value_analysis", "Not available")
    news_analysis = current_data.get("news_analysis", "Not available")

    # 处理各个分析的错误信息
    errors = []
    if "fundamental_analysis_error" in current_data:
        errors.append(
            f"Fundamental Analysis Error: {current_data['fundamental_analysis_error']}")
    if "technical_analysis_error" in current_data:
        errors.append(
            f"Technical Analysis Error: {current_data['technical_analysis_error']}")
    if "value_analysis_error" in current_data:
        errors.append(
            f"Value Analysis Error: {current_data['value_analysis_error']}")
    if "news_analysis_error" in current_data:
        errors.append(
            f"News Analysis Error: {current_data['news_analysis_error']}")

    # 基本股票标识信息
    stock_code = current_data.get("stock_code", "Unknown Stock")
    company_name = current_data.get("company_name", "Unknown Company")

    try:
        # 获取模型选择
        model_choice = get_model_choice()
        logger.info(f"{WAIT_ICON} SummaryAgent: Using model choice: {model_choice}")

        # 获取当前时间信息，用于报告中的时间标注
        current_time_info = current_data.get("current_time_info", "未知时间")
        current_date = current_data.get("current_date", "未知日期")

        # 准备汇总的系统提示词
        system_prompt = f"""
        你是一名资深的卖方金融分析师，任职于一家顶级投资银行。你的任务是为客户撰写一份专业、深入、决策导向的股票研究报告。

        **重要时间信息：**
        - **当前实际时间：{current_time_info}**
        - **报告分析基准日：{current_date}**

        这份报告必须基于真实的当前时间，所有“近期”、“历史”、“最新”等概念都要以此为准。

        **核心任务：**
        你的输入是来自四个初级分析师团队（基本面、技术、估值、新闻）的原始分析材料。你需要对这些材料进行**整合、提炼、交叉验证，并加入你自己的深刻洞见**，最终形成一份逻辑连贯、观点鲜明的高质量报告。

        **严格遵循以下报告结构和写作要求：**

        # [公司名称] ([股票代码]) 深度研究报告

        ## 1. 投资摘要 (Executive Summary)
        **目标：一分钟内让客户抓住核心观点。**
        - **核心观点**: 简明扼要地总结你对该股票的整体看法（例如，“基本面强劲，但短期技术面承压”）。
        - **量化评级**: 给出明确的评级、风险等级和预期回报。格式如下：
          - **投资评级**: 买入 / 增持 / 中性 / 减持 / 卖出
          - **风险等级**: 低 / 中低 / 中 / 中高 / 高
          - **12个月目标价**: [具体价格区间]
          - **预期回报率**: [具体百分比区间]

        ## 2. 公司概况
        **目标：介绍公司业务和护城河。**
        - 简要描述公司的核心业务、主要产品、商业模式。
        - 重点分析其在行业中的**竞争地位和核心护城河**（例如，品牌、技术、规模、渠道等）。

        ## 3. 基本面分析 (Fundamental Analysis)
        **目标：评估公司的内在健康状况。**
        - **财务实力**: 综合资产负债表的数据，评价其偿债能力（流动比率、资产负债率）和财务风险。
        - **盈利能力**: 深入分析利润表，解读**毛利率、净利率、ROE**等指标为何高/低，并与**行业平均水平**进行对比（如果数据可用）。
        - **成长性**: 结合成长能力指标，判断公司是处于高速增长、稳定增长还是成熟阶段。
        - **现金流质量**: 分析现金流量表，评价其经营活动现金流是否健康，能否覆盖投资和分红。

        ## 4. 技术分析 (Technical Analysis)
        **目标：判断市场情绪和短期走势。**
        - **价格与趋势**: 结合K线数据，清晰描述近期的主要趋势（上升/下降/震荡），并识别出关键的技术形态（如头肩顶、双底等）。
        - **量价关系**: 分析成交量的变化，判断是“放量上涨”还是“缩量下跌”，解读背后的市场含义。
        - **关键指标解读**: **必须**对移动平均线（MA）、MACD、RSI等核心指标的当前状态进行解读，说明它们发出了什么信号。
        - **支撑与阻力**: 明确给出近期的关键支撑位和阻力位。

        ## 5. 估值分析 (Valuation Analysis)
        **目标：判断当前股价是“贵”还是“便宜”。**
        - **核心估值指标**: 展示PE、PB、PS等指标。**如果数据缺失，必须明确指出**，并尝试基于已有数据（如净利润、股价）进行估算。
        - **历史估值对比**: 将当前估值与公司自身的**历史估值中枢**（例如，近5年PE百分位）进行比较。
        - **同行估值对比**: 将公司估值与**主要竞争对手或行业平均水平**进行比较（如果数据可用）。
        - **股息回报**: 计算并评价当前的**股息收益率**，并与无风险收益率（如国债收益率）进行对比。

        ## 6. 新闻与催化剂分析 (News & Catalysts)
        **目标：识别影响股价的短期事件。**
        - **近期关键新闻**: 提炼1-3条最重要的近期新闻事件。
        - **市场情绪解读**: 综合所有新闻的情感和风险评分，总结当前的市场情绪是偏乐观还是偏悲观。
        - **未来催化剂**: 预测未来可能出现的、能显著影响股价的正面或负面事件（催化剂）。

        ## 7. 综合评估与投资逻辑
        **目标：展现你的分析深度，形成投资逻辑。**
        - **一致性分析**: 找出不同分析维度之间的**相互印证**之处。例如，“基本面的强劲增长（盈利能力）在技术面上得到了放量上涨（市场认可）的验证。”
        - **矛盾点分析**: **重点分析**不同维度之间的**矛盾之处**，并给出你的解读。例如，“基本面非常优秀，但技术面却持续走弱，这可能意味着市场正在担忧未来的某个风险点...”
        - **核心投资逻辑**: 基于以上所有分析，总结出支持你最终投资建议的核心逻辑。

        ## 8. 风险因素
        **目标：全面揭示潜在风险。**
        - 分别从**宏观与行业风险**、**公司经营风险**、**财务风险**三个层面进行阐述。

        ## 9. 投资建议
        **目标：给出明确、可操作的建议。**
        - **投资评级与目标价**: 重申摘要中的评级和12个月目标价。
        - **建仓策略**: 给出具体的价格区间建议，例如“建议在XX元至XX元区间分批建仓”。
        - **适合的投资者类型**: 指明该股票适合哪类投资者（例如，长期价值投资者、成长型投资者、风险偏好者等）。

        ## 附录：数据来源与限制
        - **数据来源**: 简要说明。
        - **分析限制**: **必须**明确指出在分析过程中遇到的数据缺失或限制（例如，“本次分析未能获取行业平均估值数据，限制了横向对比的深度”）。

        **最终输出要求：**
        - 直接输出纯Markdown内容，**绝对不能**包含` ```markdown `或` ``` `。
        - 语言专业、客观、逻辑严密。
        - 在报告末尾，必须另起一行标注：“**分析基准时间：{current_time_info}**”
        
 
        """

#         ###示例###
#         # [贵州茅台](sh.600519) 综合分析报告
#
#         ## 执行摘要
#         贵州茅台作为中国高端白酒行业绝对龙头，展现出卓越的盈利能力、财务稳健性和市场竞争力。当前估值处于历史相对低位，基本面强劲支撑长期投资价值。技术面显示短期震荡整理但中长期趋势向好。近期公司在品牌建设、产品创新和人才储备方面持续发力，市场情绪整体积极。综合评估： ** 风险等级：中低 **， ** 预期回报：中长期年化10 - 15 % ** ，建议投资者现价分批建仓，适合长期价值投资者持有。
#
#         ## 公司概况
#         贵州茅台(sh
#         .600519)是中国高端白酒行业的绝对领导者，属于证监会行业分类中的C15酒、饮料和精制茶制造业。公司核心业务为茅台酒及系列酒的生产与销售，拥有极强的品牌壁垒和市场定价权。作为国内高端白酒市场的龙头企业，2024
#         年占据约40 % 的市场份额，毛利率长期维持在90 % 以上，显著高于行业平均水平。白酒行业具有典型的消费必需品属性和抗周期特征，而茅台凭借其独特的酿造工艺、稀缺产能和深厚的品牌文化，建立了难以复制的市场地位。
#
#         ## 基本面分析
#
#         ### 财务实力与资本结构
#         贵州茅台展现出极度稳健的财务结构，几乎无有息负债，资产负债率仅为0
#         .19 %，远低于行业平均水平。流动比率高达4
#         .45，现金比率1
#         .05，表明公司拥有极强的短期偿债能力和充足的现金储备。这种
#         "零负债+高现金"
#         的财务结构在A股市场极为罕见，为公司提供了极强的抗风险能力和经营灵活性。
#
#         ### 盈利能力分析
#         公司盈利能力指标处于行业顶尖水平：
#         - ** 毛利率91
#         .93 % ** ：源于茅台酒的稀缺性和高端定位，2024
#         年吨酒价格超120万元，显著高于主要竞争对手
#         - ** 净利率52
#         .27 % ** ：远超食品饮料行业平均水平(约15 - 20 %)
#         - ** 净资产收益率(ROE)
#         38.43 % ** ：连续十年维持30 % 以上，盈利效率极高
#
#         杜邦分析显示，高净利率是ROE的核心驱动因素，反映公司
#         "轻资产+高盈利"
#         的优质商业模式。2024
#         年公司营业收入达1706
#         .12
#         亿元，同比增长约15 %，净利润893
#         .35
#         亿元，同比增长15
#         .24 %，均高于高端白酒行业平均增速。
#
#         ### 现金流与运营效率
#         - ** 经营活动现金流 / 营收比率54
#         .1 % ** ：盈利质量优异，现金流覆盖能力强
#         - ** 自由现金流约925亿元 **：足以支撑高额分红与内生增长
#         - ** 应收账款周转率164
#         .5
#         次 **：近乎
#         "先款后货"
#         模式，渠道控制力极强
#         - ** 存货周转率0
#         .27
#         次 **：符合白酒行业特性(基酒需长期陈酿)，2024
#         年存货周转天数1315天(约3
#         .6
#         年)，与生产周期匹配
#
#         ### 分红政策与股东回报
#         公司实施慷慨且稳定的分红政策，2024
#         年推出中期与年度两次分红方案：
#         - 中期分红：每10股派现308
#         .76
#         元(含税)
#         - 年度分红：每10股派现238
#         .82
#         元(含税)
#         - ** 合计分红金额约696亿元 **，占净利润比例77
#         .9 %，分红率行业领先
#         - ** 股息率约3
#         .85 % ** ：高于十年期国债收益率，具备较强的防御属性
#
#         ## 技术分析
#
#         ### 价格趋势与形态
#         截至2025年8月8日，贵州茅台收盘价为10467
#         .6
#         元。近6个月股价从10342
#         .6
#         元(2
#         月10日)波动至10467
#         .6
#         元，整体呈现震荡上行趋势：
#         - ** 长期趋势(3 - 6
#         个月) ** ：2 - 3
#         月形成上升通道，4
#         月出现4
#         .39 % 回调，5
#         月反弹至阶段高点11885元，随后进入高位震荡
#         - ** 短期趋势(1
#         个月) ** ：8
#         月以来在10400 - 10550
#         元区间窄幅震荡，呈现
#         "横盘整理"
#         特征，多空双方分歧较小
#         - ** 关键形态 **：7
#         月22 - 24
#         日形成
#         "三连阳"
#         突破形态，但7月25日出现放量阴线(跌幅2
#         .45 %)，显示短期获利盘抛压
#
#         ### 成交量分析
#         - ** 量价配合 **：5
#         月14日股价跳空上涨2
#         .81 %，成交量达394
#         .6
#         万股(近期天量)，表明资金大幅流入；7
#         月25日股价大跌2
#         .45 % 时成交量同步放大至419
#         .1
#         万股，显示空头力量释放
#         - ** 近期缩量 **：8
#         月第一周成交量维持在186 - 253
#         万股，较7月平均水平(300
#         万股)明显萎缩，表明市场观望情绪浓厚
#
#         ### 技术指标分析
#
#         | 指标 | 数值 | 解读 |
#         | ------ | ------ | ------ |
#         | ** 移动平均线 ** | MA5 = 10473
#         元 | 短期均线走平，股价围绕MA5震荡，显示短期趋势不明 |
#         | | MA20 = 10526
#         元 | 中期均线略高于当前股价，形成轻微压制，需突破MA20确认反弹动能 |
#         | | MA60 = 10415
#         元 | 长期均线呈上行趋势，支撑力度较强，6
#         月以来多次在MA60附近止跌回升 |
#         | ** MACD ** | DIF = 23.5，DEA = 18.2 | DIF与DEA在零轴上方金叉后开口收窄，红柱缩短，显示多头动能减弱 |
#         | ** RSI(14) ** | 52.3 | 处于50 - 70
#         中性区间，未超买( > 70)或超卖( < 30)，短期无明显趋势信号 |
#         | ** 布林带 ** | 上轨 = 10650
#         元，中轨 = 10480
#         元，下轨 = 10310
#         元 | 股价运行于布林带中轨附近，轨道收口，预示震荡行情延续 |
#
#         ### 支撑位与阻力位
#         - ** 支撑位 **：
#         - 第一支撑：10400
#         元(近期震荡区间下沿，MA60附近)
#         - 强支撑：10123
#         元(6
#         月低点，前期密集成交区)
#         - ** 阻力位 **：
#         - 第一阻力：10550
#         元(7
#         月下旬高点，MA20压制位)
#         - 强阻力：11000
#         元(5
#         月以来的心理关口，历史套牢盘密集区)
#
#         ## 估值分析
#
#         ### 核心估值指标
#         截至估值基准日，贵州茅台主要估值指标如下：
#
#         | 指标 | 数值 | 行业对比 |
#         | ------ | ------ | ---------- |
#         | 市盈率(PE - TTM) | 20.05 | 高于中证白酒指数(14.52)，但处于自身历史低位 |
#         | 市净率(PB - MRQ) | 6.91 | 显著高于洋河股份(2.00)
#         等同行，反映品牌溢价 |
#         | 市销率(PS - TTM) | 9.97 | 高于古越龙山(4.69)，符合高毛利率特性 |
#         | 股息收益率 | 3.85 % | 高于十年期国债收益率，具备防御属性 |
#
#         ### 历史估值趋势
#         - ** PE - TTM走势(2020 - 2025) **：从2021年2月高点73
#         .29
#         倍持续回落至当前20
#         .05
#         倍，处于近5年25 % 分位以下
#         - ** PB - MRQ走势 **：从2020年的16
#         .91
#         倍回落至当前6
#         .91
#         倍，资产估值回归理性
#         - ** 当前估值评估 **：处于近5年历史低位，相对合理，已反映大部分负面预期
#
#         ### 内在价值估算
#         基于简化DCF模型估算：
#         - ** 假设条件 **：未来5年净利润增速15 % (参考历史复合增速15.2 %)，之后永续增速5 %，折现率10 %
#         - ** 估算结果 **：内在价值约每股1, 55０元
#         - ** 安全边际 **：当前股价(10467.6
#         元)较内在价值存在约9 % 的安全边际
#
#         ### 盈利能力与估值匹配度
#         公司维持行业领先的盈利能力指标：
#         - 净利润率持续高于50 % (2024年52.27 % ，2025Q1 54.89 %)
#         - ROE稳居35 % 以上(2024
#         年38
#         .43 %，2025
#         Q1年化43
#         .70 %)
#         - 经营现金流 / 净利润比率1
#         .04
#         倍，盈利质量高
#         - 高盈利能力支撑较高估值水平，PEG比率 < 1，显示估值相对合理
#
#         ## 新闻分析
#
#         ### 近期关键新闻事件
#         2025
#         年8月最新新闻显示公司在多个领域积极推进：
#
#         1. ** 飞天茅台批价稳定，红缨子高粱丰收季活动启动 ** (8月4日)
#         - 散瓶批价1860元 / 瓶，原箱1905元 / 瓶，价格体系保持稳定
#         - 丰收季活动升级为中国农民丰收节分会场，强化品牌文化内涵
#
#     2. ** 五星商标上市70周年纪念酒上线i茅台，2
#     分钟售罄 ** (8月8日)
#     - 限量25568瓶，单价7000元，销售额1
#     .79
#     亿元
#     - 同步披露2025年引进博士17人，覆盖微生物、数智化等领域，加强研发实力
#
#
# 3. ** 省级酱酒产业人才联盟启动，83
# 家单位参与 ** (8月5日)
# - 构建
# "政府+院校+企业"
# 人才培育机制，为酱酒产业集群提供人才支撑
# - 强化产业链整合能力和行业话语权
#
# ### 市场情绪与媒体报道
# - ** 正面信号(80 %) **：近期新闻主要集中在品牌营销创新、产品多元化和人才储备方面，显示公司在巩固传统优势的同时积极布局未来发展
# - ** 中性信号(20 %) **：股价短期波动属正常市场行为，未出现负面舆情
# - ** 媒体观点 **：多数财经媒体认为茅台估值已进入合理区间，长期投资价值显现
#
# ### 对股价的潜在影响
# - ** 短期(1 - 2
# 周) ** ：纪念酒售罄和丰收季活动可能提振市场信心，股价或受事件驱动小幅反弹
# - ** 中长期(6 - 12
# 个月) ** ：人才储备和产业协同将强化核心竞争力，支撑业绩持续增长
# - ** 风险点 **：需警惕宏观经济增速放缓对高端白酒需求的影响，以及行业政策(如消费税改革)
# 潜在变化
#
# ## 综合评估
#
# ### 不同分析方法的一致性
# 1. ** 基本面与估值一致性 **：
# - 强劲的基本面(高ROE、高净利率、强现金流)支撑当前估值水平
# - 历史低位的PE / PB与高股息率形成估值安全垫，基本面未出现恶化迹象
#
# 2. ** 技术面与新闻面一致性 **：
# - 技术面显示的震荡整理与新闻面反映的市场观望情绪一致
# - 正面新闻事件可能成为突破技术阻力位的催化剂
#
# ### 不同分析方法的分歧点
# 1. ** 短期趋势与长期价值 **：
# - 技术面显示短期趋势不明，处于区间震荡
# - 基本面和估值分析显示长期投资价值明确
# - ** 调和观点 **：短期波动提供建仓机会，长期持有享受企业成长红利
#
# 2. ** 估值水平争议 **：
# - 与行业平均相比PE仍较高
# - 但考虑其品牌壁垒和盈利稳定性，估值溢价合理
#
# ### 综合研判
# 贵州茅台展现出
# "高盈利、强现金流、低负债、高分红"
# 的卓越基本面，当前估值处于历史相对低位，技术面虽显示短期震荡但中长期趋势向好。公司在品牌建设、产品创新和人才储备方面的持续投入将进一步巩固其市场地位。综合来看，短期波动不改变长期投资价值，当前价位提供了较好的配置机会。
#
# ## 风险因素
#
# ### 行业风险
# 1. ** 消费需求变化 **：宏观经济增速放缓可能影响高端白酒需求
# 2. ** 行业竞争加剧 **：其他白酒企业在高端市场的竞争可能加剧
# 3. ** 消费群体变化 **：年轻一代消费者饮酒习惯改变，偏好多元化
#
# ### 公司特定风险
# 1. ** 产能天花板 **：茅台酒基酒产能约5
# .6
# 万吨 / 年，短期内难以扩张，制约营收增长潜力
# 2. ** 库存压力 **：2024
# 年渠道库存约1
# .8
# 万吨(相当于3 - 4
# 个月销量)，需警惕价格体系波动
# 3. ** 产品结构风险 **：对飞天茅台单品依赖度较高，系列酒增长不及预期
#
# ### 政策与宏观风险
# 1. ** 政策风险 **：消费税改革、公务消费限制等政策可能影响高端白酒需求
# 2. ** 原材料价格波动 **：高粱等原材料价格波动可能影响成本控制
# 3. ** 环保政策收紧 **：酿酒行业环保要求提高可能增加合规成本
#
# ### 市场风险
# 1. ** 股价波动风险 **：作为A股龙头标的，易受市场情绪和资金流动影响
# 2. ** 估值回调风险 **：若市场风格切换，高估值消费股可能面临估值压缩
# 3. ** 流动性风险 **：股价较高可能限制部分中小投资者参与，影响流动性
#
# ## 投资建议
#
# ### 投资评级与目标价格
# - ** 投资评级 **：买入
# - ** 12
# 个月目标价格 **：11500 - 12000
# 元
# - ** 预期回报 **：10 - 15 %
#
# ### 投资策略
# 1. ** 建仓策略 **：现价分批建仓，建议6 - 12
# 个月内完成目标仓位
# - 第一档：10400 - 10500
# 元区间配置40 %
# - 第二档：10000 - 10200
# 元区间配置30 %
# - 第三档：9800
# 元以下配置30 %
#
# 2. ** 持有策略 **：中长期持有(建议3年以上)，忽略短期市场波动
# - 定期检视基本面指标，非基本面恶化不轻易卖出
# - 关注季度营收增速、毛利率和库存变化等关键指标
#
# 3. ** 止盈止损 **：
# - 止损：若基本面显著恶化或股价跌破9000元(较当前价下跌14 %)
# - 止盈：达到目标价或基本面出现拐点时考虑部分止盈
#
# ### 适合投资者类型
# - ** 核心适合人群 **：长期价值投资者、收益型投资者
# - ** 风险承受能力 **：中低风险承受能力
# - ** 投资期限 **：3
# 年以上
# - ** 配置比例 **：建议在消费类资产中配置15 - 20 %
#
# ### 关键跟踪指标
# 投资者应重点关注以下指标变化：
# 1.
# 茅台酒批价走势和库存水平
# 2.
# 季度营收及净利润增速(关注是否维持15 % 左右增速)
# 3.
# 毛利率变化(关注是否维持90 % 以上)
# 4.
# 分红政策变化
# 5.
# 产能扩张计划和系列酒发展情况
#
# ## 附录：数据来源与限制
#
# ### 数据来源
# 1.
# 公司财务数据：贵州茅台2024年年报、2025
# 年一季报
# 2.
# 股价与交易数据：上海证券交易所2025年2月10日 - 8
# 月8日交易数据
# 3.
# 行业数据：中国酒业协会、Wind资讯
# 4.
# 新闻信息：公司公告、证券时报、中国证券报等权威媒体2025年8月报道
#
# ### 分析限制
# 1.
# 财务数据时效性：部分2025年最新季度数据尚未披露，分析基于已有信息
# 2.
# 估值模型简化：DCF模型采用简化假设，实际内在价值可能因假设条件变化而不同
# 3.
# 技术分析局限性：历史价格走势不代表未来表现，技术指标信号可能失效
# 4.
# 宏观环境不确定性：未考虑极端宏观经济变化对公司基本面的潜在影响
#
# ** 分析基准时间：2025
# 年08月10日(2025 - 0
# 8 - 10) 星期日
# 20: 56:01

        # 准备汇总提示词
        user_prompt = f"""
        Please create a comprehensive analysis report for {company_name} ({stock_code}) based on the following analyses.
        
        Original user query: {user_query}
        
        FUNDAMENTAL ANALYSIS:
        {fundamental_analysis}
        
        TECHNICAL ANALYSIS:
        {technical_analysis}
        
        VALUE ANALYSIS:
        {value_analysis}
        
        NEWS ANALYSIS:
        {news_analysis}
        
        {"ANALYSIS ISSUES:" if errors else ""}
        {". ".join(errors) if errors else ""}
        
        IMPORTANT: Your output MUST be in valid Markdown format with proper headings, bullet points, 
        and formatting. Include a clear recommendation section at the end.
        
        DO NOT include any code block markers like ```markdown or ``` in your output.
        Just write pure Markdown content directly.
        """

        # 根据模型选择决定使用哪种方式生成报告
        if model_choice == "local":
            # 使用本地FinR1模型
            logger.info(f"{WAIT_ICON} SummaryAgent: Using local FinR1 model...")
            
            # 记录模型配置信息
            model_config = {
                "model": "FinR1",
                "temperature": 0.5,
                "max_tokens": 5000,
                "model_path": "/home/ruan/Finance/Fin-R1"
            }

            # 加载FinR1模型
            model, tokenizer = load_finr1_model()

            # 组合完整的提示词
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            # 记录LLM交互开始时间
            llm_start_time = time.time()

            # 使用FinR1模型生成最终报告
            final_report = generate_report_with_finr1(model, tokenizer, full_prompt)

            # 记录LLM交互执行时间
            llm_execution_time = time.time() - llm_start_time

        else:
            # 默认使用API接口
            logger.info(f"{WAIT_ICON} SummaryAgent: Using OpenAI API...")
            
            # 创建OpenAI模型（使用直接API调用，而不是ReAct框架进行汇总）
            api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
            base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
            model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")

            # 验证必要的环境变量是否存在
            if not all([api_key, base_url, model_name]):
                logger.error(
                    f"{ERROR_ICON} SummaryAgent: Missing OpenAI environment variables.")
                current_data["summary_error"] = "Missing OpenAI environment variables."

                # 记录 Agent执行失败
                execution_logger.log_agent_complete(agent_name, current_data, time.time(
                ) - agent_start_time, False, "Missing OpenAI environment variables")

                return {"data": current_data, "messages": messages}
            model_name = 'gpt-4o'
            # 记录模型配置信息
            model_config = {
                "model": model_name,
                "temperature": 0.5,
                "max_tokens": 5000,
                "api_base": base_url
            }

            # 准备汇总提示词消息列表
            summary_prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            ###vllm
            if model_choice == 'vllm':
                model_name = "./Fin-R1-awq"
                base_url = "http://localhost:8000/v1"

            # 使用ChatOpenAI模型
            logger.info(f"{WAIT_ICON} SummaryAgent: Creating ChatOpenAI with model {model_name}")
            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=0.5,  # 提高温度以增加创造性和更自然的表达
                max_tokens=8000   # 增大输出长度以生成更详细的综合报告
            )

            # 记录LLM交互开始时间
            llm_start_time = time.time()

            # 调用LLM生成最终报告
            llm_message = await llm.ainvoke(summary_prompt_messages)
            final_report = llm_message.content

            # 记录LLM交互执行时间
            llm_execution_time = time.time() - llm_start_time

        # 记录LLM交互详情，用于后续分析和优化
        execution_logger.log_llm_interaction(
            agent_name=agent_name,
            interaction_type="summary_generation",
            input_messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            output_content=final_report,
            model_config=model_config,
            execution_time=llm_execution_time
        )

        # 移除任何可能出现的markdown代码块标记
        final_report = final_report.replace(
            "```markdown", "").replace("```", "").strip()
        
        # 使用正则表达式截断"分析基准时间"那一行之后的内容
        final_report = truncate_report_at_baseline_time(final_report, current_time_info)

        logger.info(
            f"{SUCCESS_ICON} SummaryAgent: Final report generated for {company_name} ({stock_code}).")
        logger.debug(f"Final report preview: {final_report[:300]}...")

        # 将报告保存到Markdown文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 处理公司名称和股票代码，确保文件名有意义
        if stock_code == "Unknown Stock" or stock_code == "Extracted from analysis":
            # 从用户查询中提取更有意义的名称
            query_based_name = user_query.replace(
                " ", "_").replace("分析", "").strip()
            if not query_based_name:
                query_based_name = "financial_analysis"
            safe_file_prefix = f"report_{query_based_name}"
        else:
            # 正常情况下使用公司名称和股票代码
            safe_company_name = company_name.replace(" ", "_").replace(".", "")
            if safe_company_name == "Unknown_Company" or safe_company_name == "Extracted_from_analysis":
                safe_company_name = user_query.replace(
                    " ", "_").replace("分析", "").strip()
                if not safe_company_name:
                    safe_company_name = "company"

            # 清理股票代码（移除可能的前缀）
            clean_stock_code = stock_code.replace("sh.", "").replace("sz.", "")
            safe_file_prefix = f"report_{safe_company_name}_{clean_stock_code}"

        report_filename = f"{safe_file_prefix}_{timestamp}.md"

        # 确保reports目录存在
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "reports")
        os.makedirs(reports_dir, exist_ok=True)

        report_path = os.path.join(reports_dir, report_filename)

        # 将报告写入文件
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report)

        logger.info(
            f"{SUCCESS_ICON} SummaryAgent: Report saved to {report_path}")

        # 返回更新后的状态，包含最终报告
        current_data["final_report"] = final_report
        current_data["report_path"] = report_path

        # 记录 Agent执行成功
        total_execution_time = time.time() - agent_start_time
        execution_logger.log_agent_complete(agent_name, {
            "final_report_length": len(final_report),
            "report_path": report_path,
            "report_preview": final_report,
            "llm_execution_time": llm_execution_time,
            "total_execution_time": total_execution_time
        }, total_execution_time, True)

        return {"data": current_data, "messages": messages}

    except Exception as e:
        logger.error(
            f"{ERROR_ICON} SummaryAgent: Error generating final report: {e}", exc_info=True)
        current_data["summary_error"] = f"Error generating final report: {e}"

        # 即使出现错误也创建最小化的报告
        error_report = f"""
        # Analysis Report for {company_name} ({stock_code})
        
        **Error encountered during report generation**: {e}
        
        ## Available Analysis Fragments:
        
        - Fundamental Analysis: {"Available" if fundamental_analysis != "Not available" else "Not available"}
        - Technical Analysis: {"Available" if technical_analysis != "Not available" else "Not available"}
        - Value Analysis: {"Available" if value_analysis != "Not available" else "Not available"}
        - News Analysis: {"Available" if news_analysis != "Not available" else "Not available"}
        
        Please review the individual analyses directly for more information.
        """
        current_data["final_report"] = error_report

        # 也将错误报告保存到文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 处理公司名称和股票代码，确保文件名有意义
        if stock_code == "Unknown Stock" or stock_code == "Extracted from analysis":
            # 从用户查询中提取更有意义的名称
            query_based_name = user_query.replace(
                " ", "_").replace("分析", "").strip()
            if not query_based_name:
                query_based_name = "financial_analysis"
            safe_file_prefix = f"error_report_{query_based_name}"
        else:
            # 正常情况下使用公司名称和股票代码
            safe_company_name = company_name.replace(" ", "_").replace(".", "")
            if safe_company_name == "Unknown_Company" or safe_company_name == "Extracted_from_analysis":
                safe_company_name = user_query.replace(
                    " ", "_").replace("分析", "").strip()
                if not safe_company_name:
                    safe_company_name = "company"

            # 清理股票代码（移除可能的前缀）
            clean_stock_code = stock_code.replace("sh.", "").replace("sz.", "")
            safe_file_prefix = f"error_report_{safe_company_name}_{clean_stock_code}"

        report_filename = f"{safe_file_prefix}_{timestamp}.md"

        # 确保reports目录存在
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "reports")
        os.makedirs(reports_dir, exist_ok=True)

        report_path = os.path.join(reports_dir, report_filename)

        # 将错误报告写入文件
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(error_report)

        logger.info(
            f"{ERROR_ICON} SummaryAgent: Error report saved to {report_path}")
        current_data["report_path"] = report_path

        # 记录 Agent执行失败
        execution_logger.log_agent_complete(
            agent_name, current_data, time.time() - agent_start_time, False, str(e))

        return {"data": current_data, "messages": messages}


# 本地测试函数
async def test_summary_agent():
    """汇总 Agent的测试函数"""
    from src.utils.state_definition import AgentState

    # 用于测试的示例状态，包含模拟分析结果
    test_state = AgentState(
        messages=[],
        data={
            "query": "分析嘉友国际",
            "stock_code": "603871",
            "company_name": "嘉友国际",
            "fundamental_analysis": "嘉友国际基本面分析：公司主营业务为跨境物流、供应链贸易以及供应链增值服务。财务状况良好，负债率较低，现金流充裕。近年来业绩稳步增长，毛利率保持在行业较高水平。",
            "technical_analysis": "嘉友国际技术分析：短期内股价处于上升通道，突破了200日均线。RSI指标显示股票尚未达到超买区域。MACD指标呈现多头形态，成交量有所放大，支持价格继续上行。",
            "value_analysis": "嘉友国际估值分析：当前市盈率为15倍，低于行业平均水平。市净率为1.8倍，处于合理区间。与同行业公司相比，嘉友国际的估值较为合理，具有一定的投资价值。",
            "news_analysis": "嘉友国际新闻分析：近期公司发布了2023年业绩预告，预计净利润同比增长15-25%，超出市场预期。同时，公司宣布与多家国际物流巨头达成战略合作，市场反应积极。分析师普遍上调了目标价，市场情绪偏向乐观。"
        },
        metadata={}
    )

    # 运行 Agent并输出结果
    result = await summary_agent(test_state)
    print("Summary Report:")
    print(result.get("data", {}).get("final_report", "No report generated"))
    print(
        f"Report saved to: {result.get('data', {}).get('report_path', 'Not saved')}")

    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_summary_agent())
