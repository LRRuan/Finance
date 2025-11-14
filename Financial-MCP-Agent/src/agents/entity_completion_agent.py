"""
EntityCompletionAgent: Uses a ReAct Agent to extract and complete stock entities (company name and stock code).
实体补全Agent：使用ReAct Agent来提取并补全股票实体（公司名称和股票代码）。
"""
import os
import json
import re
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import time

from src.utils.state_definition import AgentState
from src.tools.mcp_client import get_mcp_tools
from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from dotenv import load_dotenv

load_dotenv(override=True)
logger = setup_logger(__name__)


# 用于从LLM的最终输出中解析出JSON
def parse_final_answer_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.error(f"无法从最终答案中解析JSON: {text}")
            return None
    return None


async def entity_completion_agent(state: AgentState) -> AgentState:
    """
    使用ReAct Agent和工具来提取并补全公司名称和股票代码。
    """
    logger.info(f"{WAIT_ICON} EntityCompletionAgent: Starting entity extraction and completion.")

    # 状态和日志记录器初始化
    current_data = state.get("data", {})
    user_query = current_data.get("query")
    execution_logger = get_execution_logger()
    agent_name = "entity_completion_agent"
    agent_start_time = time.time()

    execution_logger.log_agent_start(agent_name, {"user_query": user_query})

    if not user_query:
        error_msg = "User query is missing."
        logger.error(f"{ERROR_ICON} EntityCompletionAgent: {error_msg}")
        current_data["entity_extraction_error"] = error_msg
        execution_logger.log_agent_complete(agent_name, {}, 0, False, error_msg)
        return {"data": current_data}

    try:
        # 1. 创建LLM实例
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_COMPATIBLE_MODEL"),
            api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY"),
            base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL"),
            temperature=0.0,  # 对于提取任务，温度设为0最稳定
        )

        # 2. 获取工具，但只筛选出新闻搜索工具
        logger.info(f"{WAIT_ICON} EntityCompletionAgent: Fetching tools...")
        all_tools = await get_mcp_tools()
        # 我们只需要 crawl_news 和 get_stock_basic_info 这两个工具
        # Tavily的crawl_news善于通过名字找信息，baostock的get_stock_basic_info善于通过代码找名字
        completion_tools = [t for t in all_tools if t.name in ["web_search", "get_stock_basic_info"]]

        if not completion_tools:
            error_msg = "Required tools ('web_search', 'get_stock_basic_info') not found."
            raise ValueError(error_msg)

        logger.info(f"{SUCCESS_ICON} EntityCompletionAgent: Using tools: {[t.name for t in completion_tools]}")

        # 3. 创建ReAct Agent
        agent_executor = create_react_agent(llm, completion_tools)

        # 4. 构建一个非常精确和强大的Prompt
        prompt = f"""
        你的任务是从用户的查询中识别出公司名称或股票代码，并确保两者都被补全。

        用户查询: "{user_query}"

        你的思考过程和行动步骤应该是：
        1.  **识别**: 直接从查询中识别出公司名称或股票代码。
        2.  **补全**:
            *   如果你只识别出了**公司名称**，你必须使用 `web_search` 工具搜索 `[公司名称] 股票代码` 来找到对应的6位数字股票代码。
            *   如果你只识别出了**股票代码**，你必须使用 `get_stock_basic_info` 工具来查询这个代码，并从返回结果中找到对应的公司名称 (`code_name`)。
            *   如果两者都已经存在，验证它们是否匹配。
        3.  **格式化**: 在完成所有步骤后，你必须以一个最终的、纯粹的JSON格式对象作为你的Final Answer，绝对不能包含任何其他解释性文字。JSON格式如下：
            {{"company_name": "补全后的公司名称", "stock_code": "补全后的6位数字股票代码"}}

        现在，开始你的工作。
        """

        # 5. 调用Agent
        logger.info(f"{WAIT_ICON} EntityCompletionAgent: Invoking ReAct agent...")
        response = await agent_executor.ainvoke({"messages": [HumanMessage(content=prompt)]})

        # 6. 解析最终结果
        final_answer = response['messages'][-1].content
        logger.info(f"ReAct agent final answer: {final_answer}")

        parsed_data = parse_final_answer_json(final_answer)

        if not parsed_data or not parsed_data.get("company_name") or not parsed_data.get("stock_code"):
            raise ValueError(f"Agent未能成功补全实体，最终答案: {final_answer}")

        company_name = parsed_data["company_name"]
        stock_code = parsed_data["stock_code"]

        # 7. 更新状态
        current_data["company_name"] = company_name

        # 自动添加 'sh.' 或 'sz.' 前缀
        if stock_code.startswith('6'):
            current_data["stock_code"] = f"sh.{stock_code}"
        elif stock_code.startswith(('0', '3')):
            current_data["stock_code"] = f"sz.{stock_code}"
        else:
            current_data["stock_code"] = stock_code  # 保持原样以防其他市场

        logger.info(f"{SUCCESS_ICON} EntityCompletionAgent: Entities successfully completed. "
                    f"Company Name='{current_data['company_name']}', Stock Code='{current_data['stock_code']}'")

        execution_time = time.time() - agent_start_time
        execution_logger.log_agent_complete(agent_name, {"company_name": company_name, "stock_code": stock_code},
                                            execution_time, True)

        return {"data": current_data}

    except Exception as e:
        error_msg = f"Error during entity completion: {e}"
        logger.error(f"{ERROR_ICON} EntityCompletionAgent: {error_msg}", exc_info=True)
        current_data["entity_extraction_error"] = error_msg
        execution_time = time.time() - agent_start_time
        execution_logger.log_agent_complete(agent_name, {}, execution_time, False, error_msg)
        return {"data": current_data}