# Main MCP server file
import logging
from datetime import datetime

# from mcp.server.fastmcp import FastMCP
from fastmcp import FastMCP
import baostock as bs

# Import the interface and the concrete implementation
from src.data_source_interface import FinancialDataSource
from src.baostock_data_source import BaostockDataSource
from src.utils import setup_logging

# 导入各模块工具的注册函数
from src.tools.stock_market import register_stock_market_tools
from src.tools.financial_reports import register_financial_report_tools
from src.tools.indices import register_index_tools
from src.tools.market_overview import register_market_overview_tools
from src.tools.macroeconomic import register_macroeconomic_tools
from src.tools.date_utils import register_date_utils_tools
from src.tools.analysis import register_analysis_tools
from src.tools.news_crawler import register_news_crawler_tools

from dotenv import load_dotenv
import atexit

# --- Logging Setup ---
# Call the setup function from utils
# You can control the default level here (e.g., logging.DEBUG for more verbose logs)
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Dependency Injection ---
# Instantiate the data source - easy to swap later if needed
active_data_source: FinancialDataSource = BaostockDataSource()

# --- Get current date for system prompt ---
current_date = datetime.now().strftime("%Y-%m-%d")

# --- FastMCP App Initialization ---
app = FastMCP(
#     server_name="a_share_data_provider",
#     description=f"""今天是{current_date}。提供中国A股市场数据分析工具。此服务提供客观数据分析，用户需自行做出投资决策。数据分析基于公开市场信息，不构成投资建议，仅供参考。

# ⚠️ 重要说明:
# 1. 最新交易日不一定是今天，需要从 get_latest_trading_date() 获取
# 2. 请始终使用 get_latest_trading_date() 工具获取实际当前最近的交易日，不要依赖训练数据中的日期认知
# 3. 当分析"最近"或"近期"市场情况时，必须首先调用 get_market_analysis_timeframe() 工具确定实际的分析时间范围
# 4. 任何涉及日期的分析必须基于工具返回的实际数据，不得使用过时或假设的日期
# 5. 新增新闻爬虫功能，可以搜索公司、行业相关新闻，辅助投资决策
# """,
    # Specify dependencies for installation if needed (e.g., when using `mcp install`)
    # dependencies=["baostock", "pandas"]
)

# --- 注册各模块的工具 ---
register_stock_market_tools(app, active_data_source)
register_financial_report_tools(app, active_data_source)
register_index_tools(app, active_data_source)
register_market_overview_tools(app, active_data_source)
register_macroeconomic_tools(app, active_data_source)
register_date_utils_tools(app, active_data_source)
register_analysis_tools(app, active_data_source)
register_news_crawler_tools(app, active_data_source)


def silent_login():
    """
    执行Baostock登录操作，但不向标准输出打印任何信息，以避免干扰stdio通信。
    返回登录结果对象。
    """
    import sys
    import os
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        lg = bs.login()
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout
    return lg


def silent_logout():
    """
    执行Baostock登出操作，但不向标准输出打印任何信息，以避免干扰stdio通信。
    """
    # 我们可以通过临时重定向stdout来实现静默
    import sys
    import os
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        bs.logout()
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout
        logging.getLogger(__name__).info("✓ Baostock global session silently logged out.")


if __name__ == "__main__":
    # load_dotenv()
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # logger = logging.getLogger(__name__)
    #
    logger.info("--- MCP Server Starting ---")
    #
    # # --- 全局 Baostock 登录逻辑 ---
    logger.info("Attempting global Baostock login...")
    # 使用silent_login来抑制baostock库的输出，避免干扰MCP的stdio通信
    lg = silent_login()
    if lg.error_code == '0':
        logger.info("✓ Baostock global login successful.")
        # 注册我们自定义的"安静"登出函数
        atexit.register(silent_logout)
        logger.info("Baostock silent logout hook registered.")
    else:
        logger.error(f"✗ Baostock global login FAILED: {lg.error_msg}. Server will exit.")
        exit(1)
    # Run the server using stdio transport, suitable for MCP Hosts like Claude Desktop
    # app.run(transport='stdio')
    app.run(transport='http',
        # host='0.0.0.0',    # 2. 监听所有网络接口，允许来自本机和其他机器的连接
        port=8080          # 3. 指定监听的端口号
    )