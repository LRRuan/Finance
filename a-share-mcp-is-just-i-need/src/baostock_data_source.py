# 使用Baostock库实现FinancialDataSource接口的具体数据源
import baostock as bs  # Baostock数据API库，用于获取A股市场数据
import pandas as pd    # 数据处理和分析库，用于处理返回的数据框
from typing import List, Optional, Dict  # 类型注解支持，增强代码可读性和类型检查
import logging         # 日志记录模块，用于跟踪程序执行和调试
from .data_source_interface import FinancialDataSource, DataSourceError, NoDataFoundError, LoginError
from .utils import (
    baostock_login_context,  # 登录上下文管理器，自动处理登录登出
    fetch_financial_data,    # 通用财务数据获取函数
    fetch_index_constituent_data,  # 通用指数成分股数据获取函数
    fetch_macro_data,        # 通用宏观经济数据获取函数
    fetch_generic_data,      # 通用数据获取函数
    format_fields            # 字段格式化函数
)
import os
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import json
from dotenv import load_dotenv
from transformers import BitsAndBytesConfig

load_dotenv(override=True)
# 为当前模块创建专用的日志记录器，便于调试和错误追踪
logger = logging.getLogger(__name__)

# K线数据的默认字段，包含股票的基本交易信息和财务指标
DEFAULT_K_FIELDS = [
    "date",        # 交易日期
    "code",        # 股票代码
    "open",        # 开盘价
    "high",        # 最高价
    "low",         # 最低价
    "close",       # 收盘价
    "preclose",    # 前收盘价
    "volume",      # 成交量
    "amount",      # 成交金额
    "adjustflag",  # 复权类型标识
    "turn",        # 换手率
    "tradestatus", # 交易状态
    "pctChg",      # 涨跌幅
    "peTTM",       # 市盈率TTM
    "pbMRQ",       # 市净率MRQ
    "psTTM",       # 市销率TTM
    "pcfNcfTTM",   # 市现率TTM
    "isST"         # 是否ST股票
]

# 股票基本信息的默认字段
DEFAULT_BASIC_FIELDS = [
    "code",        # 股票代码
    "tradeStatus", # 交易状态
    "code_name"    # 股票名称
    # 可根据需要添加更多默认字段，如"industry"(行业), "listingDate"(上市日期)
]


class BaostockDataSource(FinancialDataSource):
    """
    使用Baostock library实现FinancialDataSource接口的实现类
    """

    def _format_fields(self, fields: Optional[List[str]], default_fields: List[str]) -> str:
        """
        将字段列表格式化为Baostock API所需的逗号分隔字符串
        
        参数:
            fields: 用户请求的字段列表（可选）
            default_fields: 默认字段列表（当fields为空时使用）
            
        返回:
            逗号分隔的字段字符串
            
        异常:
            ValueError: 如果请求的字段包含非字符串类型
        """
        return format_fields(fields, default_fields)

    def get_profit_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """使用Baostock获取季度盈利能力数据"""
        return fetch_financial_data(bs.query_profit_data, "Profitability", code, year, quarter)

    def get_operation_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """使用Baostock获取季度运营能力数据"""
        return fetch_financial_data(bs.query_operation_data, "Operation Capability", code, year, quarter)

    def get_growth_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """使用Baostock获取季度成长能力数据"""
        return fetch_financial_data(bs.query_growth_data, "Growth Capability", code, year, quarter)

    def get_balance_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """使用Baostock获取季度资产负债表数据（偿债能力）"""
        return fetch_financial_data(bs.query_balance_data, "Balance Sheet", code, year, quarter)

    def get_cash_flow_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """使用Baostock获取季度现金流量数据"""
        return fetch_financial_data(bs.query_cash_flow_data, "Cash Flow", code, year, quarter)

    def get_dupont_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        """使用Baostock获取季度杜邦分析数据"""
        return fetch_financial_data(bs.query_dupont_data, "DuPont Analysis", code, year, quarter)

    def get_sz50_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        """使用Baostock获取深证50指数成分股"""
        return fetch_index_constituent_data(bs.query_sz50_stocks, "SZSE 50", date)

    def get_hs300_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        """使用Baostock获取沪深300指数成分股"""
        return fetch_index_constituent_data(bs.query_hs300_stocks, "CSI 300", date)

    def get_zz500_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        """使用Baostock获取中证500指数成分股"""
        return fetch_index_constituent_data(bs.query_zz500_stocks, "CSI 500", date)

    def get_deposit_rate_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """使用Baostock获取基准存款利率"""
        return fetch_macro_data(bs.query_deposit_rate_data, "Deposit Rate", start_date, end_date)

    def get_loan_rate_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """使用Baostock获取基准贷款利率"""
        return fetch_macro_data(bs.query_loan_rate_data, "Loan Rate", start_date, end_date)

    def get_required_reserve_ratio_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None, year_type: str = '0') -> pd.DataFrame:
        """使用Baostock获取存款准备金率数据"""
        # 注意额外的yearType参数通过kwargs处理
        return fetch_macro_data(bs.query_required_reserve_ratio_data, "Required Reserve Ratio", start_date, end_date, yearType=year_type)

    def get_money_supply_data_month(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """使用Baostock获取月度货币供应量数据（M0、M1、M2）"""
        # Baostock期望这里的日期格式为YYYY-MM
        return fetch_macro_data(bs.query_money_supply_data_month, "Monthly Money Supply", start_date, end_date)

    def get_money_supply_data_year(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """使用Baostock获取年度货币供应量数据（M0、M1、M2 - 年末余额）"""
        # Baostock期望这里的日期格式为YYYY
        return fetch_macro_data(bs.query_money_supply_data_year, "Yearly Money Supply", start_date, end_date)

    def get_trade_dates(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """获取指定时间范围内的交易日历数据"""
        return fetch_macro_data(bs.query_trade_dates, "Trade Dates", start_date, end_date)

    def get_historical_k_data(
        self,
        code: str,                    # 股票代码（如"sh.600000"）
        start_date: str,              # 开始日期（如"2023-01-01"）
        end_date: str,                # 结束日期（如"2023-12-31"）
        frequency: str = "d",         # 数据频率：d=日线，w=周线，m=月线
        adjust_flag: str = "3",       # 复权类型：1=前复权，2=后复权，3=不复权
        fields: Optional[List[str]] = None,  # 可选字段列表
    ) -> pd.DataFrame:
        """获取股票历史K线数据"""
        logger.info(
            f"Fetching K-data for {code} ({start_date} to {end_date}), freq={frequency}, adjust={adjust_flag}")
        
        try:
            # 格式化请求字段，如果未指定则使用默认K线字段
            formatted_fields = self._format_fields(fields, DEFAULT_K_FIELDS)

            # 关键判断：如果频率是周线或月线，就从字段中移除 'preclose'
            if frequency in ['w', 'm']:
                field_list = formatted_fields.split(',')
                # 把所有已知的、在周线/月线下不支持的字段都列在这里
                unsupported_fields = ['preclose', 'tradestatus']

                original_fields_set = set(field_list)

                # 使用列表推导式安全地移除所有不支持的字段
                field_list = [f for f in field_list if f not in unsupported_fields]

                removed_fields = original_fields_set - set(field_list)
                if removed_fields:
                    logger.info(f"Weekly/Monthly query detected. Removed unsupported fields: {list(removed_fields)}.")

                formatted_fields = ','.join(field_list)


            # 使用登录上下文管理器确保API连接
            with baostock_login_context():
                # 调用Baostock API获取K线数据
                rs = bs.query_history_k_data_plus(
                    code,
                    formatted_fields,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency,
                    adjustflag=adjust_flag
                )

                # 检查API返回的错误码
                if rs.error_code != '0':
                    logger.error(
                        f"Baostock API error (K-data) for {code}: {rs.error_msg} (code: {rs.error_code})")

                    # 区分"无数据"和"API错误"两种情况
                    if "no record found" in rs.error_msg.lower() or rs.error_code == '10002':
                        raise NoDataFoundError(
                            f"No historical data found for {code} in the specified range. Baostock msg: {rs.error_msg}")
                    else:
                        raise DataSourceError(
                            f"Baostock API error fetching K-data: {rs.error_msg} (code: {rs.error_code})")

                # 遍历结果集，收集所有数据行
                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                # 检查是否为空结果集
                if not data_list:
                    logger.warning(
                        f"No historical data found for {code} in range (empty result set from Baostock).")
                    raise NoDataFoundError(
                        f"No historical data found for {code} in the specified range (empty result set).")

                # 将数据转换为DataFrame，使用API返回的字段名作为列名
                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(f"Retrieved {len(result_df)} records for {code}.")
                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            # 已知异常直接重新抛出
            logger.warning(
                f"Caught known error fetching K-data for {code}: {type(e).__name__}")
            raise e
        except Exception as e:
            # 未知异常包装为DataSourceError
            logger.exception(
                f"Unexpected error fetching K-data for {code}: {e}")
            raise DataSourceError(
                f"Unexpected error fetching K-data for {code}: {e}")

    def get_stock_basic_info(self, code: str, fields: Optional[List[str]] = None) -> pd.DataFrame:
        """获取股票基本信息（如股票名称、交易状态等）"""
        logger.info(f"Fetching basic info for {code}")
        
        try:
            # 记录调试信息：请求的字段
            logger.debug(
                f"Requesting basic info for {code}. Optional fields requested: {fields}")

            # 使用登录上下文管理器
            with baostock_login_context():
                # 调用Baostock API获取股票基本信息
                rs = bs.query_stock_basic(code=code)

                # 检查API错误
                if rs.error_code != '0':
                    logger.error(
                        f"Baostock API error (Basic Info) for {code}: {rs.error_msg} (code: {rs.error_code})")

                    # 区分无数据和API错误
                    if "no record found" in rs.error_msg.lower() or rs.error_code == '10002':
                        raise NoDataFoundError(
                            f"No basic info found for {code}. Baostock msg: {rs.error_msg}")
                    else:
                        raise DataSourceError(
                            f"Baostock API error fetching basic info: {rs.error_msg} (code: {rs.error_code})")

                # 收集数据行
                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())

                # 检查空结果
                if not data_list:
                    logger.warning(
                        f"No basic info found for {code} (empty result set from Baostock).")
                    raise NoDataFoundError(
                        f"No basic info found for {code} (empty result set).")

                # 转换为DataFrame
                result_df = pd.DataFrame(data_list, columns=rs.fields)
                logger.info(
                    f"Retrieved basic info for {code}. Columns: {result_df.columns.tolist()}")

                # 如果用户指定了字段，则筛选返回的列
                if fields:
                    # 找出用户请求的字段中实际存在的列
                    available_cols = [
                        col for col in fields if col in result_df.columns]

                    # 如果用户请求的字段都不存在，则报错
                    if not available_cols:
                        raise ValueError(
                            f"None of the requested fields {fields} are available in the basic info result.")

                    logger.debug(
                        f"Selecting columns: {available_cols} from basic info for {code}")
                    result_df = result_df[available_cols]

                return result_df

        except (LoginError, NoDataFoundError, DataSourceError, ValueError) as e:
            # 已知异常重新抛出
            logger.warning(
                f"Caught known error fetching basic info for {code}: {type(e).__name__}")
            raise e
        except Exception as e:
            # 未知异常包装
            logger.exception(
                f"Unexpected error fetching basic info for {code}: {e}")
            raise DataSourceError(
                f"Unexpected error fetching basic info for {code}: {e}")

    def get_dividend_data(self, code: str, year: str, year_type: str = "report") -> pd.DataFrame:
        """获取股票分红派息数据"""
        return fetch_generic_data(
            bs.query_dividend_data,
            "Dividend",
            code=code,
            year=year,
            yearType=year_type
        )

    def get_adjust_factor_data(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票复权因子数据"""
        return fetch_generic_data(
            bs.query_adjust_factor,
            "Adjustment Factor",
            code=code,
            start_date=start_date,
            end_date=end_date
        )

    def get_performance_express_report(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票业绩快报数据（业绩快报）"""
        return fetch_generic_data(
            bs.query_performance_express_report,
            "Performance Express Report",
            code=code,
            start_date=start_date,
            end_date=end_date
        )

    def get_forecast_report(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票业绩预告数据（业绩预告）"""
        return fetch_generic_data(
            bs.query_forecast_report,
            "Performance Forecast Report",
            code=code,
            start_date=start_date,
            end_date=end_date
        )

    def get_stock_industry(self, code: Optional[str] = None, date: Optional[str] = None) -> pd.DataFrame:
        """获取股票行业分类数据"""
        return fetch_generic_data(
            bs.query_stock_industry,
            "Industry",
            code=code,
            date=date
        )

    def get_all_stock(self, date: Optional[str] = None) -> pd.DataFrame:
        """获取指定日期的全市场股票列表"""
        return fetch_generic_data(
            bs.query_all_stock,
            "All Stock List",
            day=date
        )

    def web_search(self, query: str, max_results: int = 3) -> str:
        """
        使用 Tavily API 进行通用的网络搜索，返回一个简洁的结果列表。
        这个工具不获取全文，也不进行情感/风险分析，只用于快速查找信息。

        Args:
            query: 搜索查询词，例如 "宁德时代 股票代码"。
            max_results: 返回结果的最大数量。

        Returns:
            一个包含搜索结果的JSON字符串。
        """
        logger.info("Initializing BaostockDataSource...")
        try:
            # 从环境变量中获取Tavily API Key并初始化客户端
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                logger.warning(
                    "TAVILY_API_KEY apy key not found in environment variables. News crawling will be disabled.")
                self.tavily_client = None
            else:
                self.tavily_client = TavilyClient(api_key=api_key)
                logger.info("TavilyClient initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize TavilyClient: {e}")
            self.tavily_client = None

        if not self.tavily_client:
            return json.dumps({"error": "Tavily Client 未初始化，无法进行网络搜索。"})

        logger.info(f"执行通用网络搜索 (web_search)，查询词: '{query}'")
        try:
            # 使用 Tavily 的基础搜索功能
            response = self.tavily_client.search(
                query=query,
                search_depth="basic",  # "basic" 模式更快，只返回摘要
                max_results=max_results
            )

            # 只提取对实体补全有用的核心信息
            results = [
                {
                    "title": res.get("title"),
                    "url": res.get("url"),
                    "content": res.get("content")  # 这是摘要，不是全文
                }
                for res in response.get("results", [])
            ]

            return json.dumps(results, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"执行 web_search 时出错: {e}")
            return json.dumps({"error": f"网络搜索时发生错误: {str(e)}"})
    # 新增爬虫功能
    def crawl_news(self, query: str, top_k: int = 10) -> str:
        """
                使用 Tavily 搜索引擎 API 获取相关新闻，并使用本地模型进行风险和情感分析。

                Args:
                    query: 核心搜索查询词 (例如公司名称 "宁德时代")
                    top_k: 需要返回和分析的新闻数量

                Returns:
                    格式化的新闻分析结果JSON字符串，或错误信息。
        """

        logger.info("Initializing BaostockDataSource...")
        try:
            # 从环境变量中获取Tavily API Key并初始化客户端
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                logger.warning(
                    "TAVILY_API_KEY apy key not found in environment variables. News crawling will be disabled.")
                self.tavily_client = None
            else:
                self.tavily_client = TavilyClient(api_key=api_key)
                logger.info("TavilyClient initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize TavilyClient: {e}")
            self.tavily_client = None

        if not self.tavily_client:
            return json.dumps({"error": "Tavily Client 未配置或初始化失败，无法搜索新闻。"})

        logger.info(f"开始使用 Tavily 搜索新闻，查询词: '{query}', 数量: {top_k}")

        try:
            # 优化搜索查询，使其更具体
            search_query = f"关于'{query}'的最新公司财务新闻、重大公告或业务进展"

            response = self.tavily_client.search(
                query=search_query,
                search_depth="advanced",
                max_results=top_k,
                include_raw_content=True  # Tavily会帮我们获取原始内容
            )

            search_results = response.get('results', [])
            if not search_results:
                logger.warning(f"Tavily 未找到关于 '{query}' 的相关新闻。")
                return json.dumps({"message": f"未找到关于 '{query}' 的相关新闻。"})

            # 加载分析模型
            risk_model, risk_tokenizer = self._load_risk_model()
            sentiment_model, sentiment_tokenizer = self._load_sentiment_model()

            # 如果模型加载失败，优雅地处理
            if not risk_model or not sentiment_model:
                return json.dumps({"error": "风险或情感分析模型加载失败，无法完成分析。"})

            analyzed_results = []
            for result in search_results:
                try:
                    content = result.get('raw_content', '')
                    # 只分析有足够内容的文章
                    if content and len(content) > 150:
                        risk_analysis = self._analyze_risk(content, risk_model, risk_tokenizer)
                        sentiment_analysis = self._analyze_sentiment(content, sentiment_model, sentiment_tokenizer)

                        analyzed_results.append({
                            'title': result.get('title', '无标题'),
                            'link': result.get('url', ''),
                            'source': result.get('source', '未知来源'),
                            'risk': risk_analysis,
                            'sentiment': sentiment_analysis
                        })
                except Exception as e:
                    logger.warning(f"分析单条新闻 '{result.get('title')}' 时出错: {e}")
                    continue

            if not analyzed_results:
                return json.dumps({"message": "虽然找到了相关新闻标题，但无法获取足够内容进行分析。"})

            # 以JSON格式返回，对Agent更友好
            return json.dumps(analyzed_results, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"使用 Tavily 爬取和分析新闻时发生严重错误: {e}")
            return json.dumps({"error": f"爬取新闻时出错: {str(e)}"})



        # """
        # 直接从浏览器搜索并爬取相关文章内容，并使用风险模型和情感模型进行分析
        #
        # Args:
        #     query: 用户查询
        #     top_k: 返回的新闻数量
        #
        # Returns:
        #     格式化的新闻结果
        # """
        # try:
        #
        #
        #     # 加载风险模型和情感模型
        #     risk_model, risk_tokenizer = self._load_risk_model()
        #     sentiment_model, sentiment_tokenizer = self._load_sentiment_model()
        #
        #     # 使用百度搜索
        #     search_url = f"https://www.baidu.com/s?wd={query}"
        #     headers = {
        #         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        #     }
        #
        #     response = requests.get(search_url, headers=headers, timeout=10)
        #     response.raise_for_status()
        #
        #     soup = BeautifulSoup(response.content, 'html.parser')
        #
        #     # 提取搜索结果
        #     results = []
        #     search_results = soup.find_all('div', class_='result')
        #
        #     for result in search_results[:top_k]:
        #         try:
        #             # 提取标题和链接
        #             title_elem = result.find('h3')
        #             if title_elem:
        #                 title = title_elem.get_text(strip=True)
        #                 link = title_elem.find('a')['href'] if title_elem.find('a') else ''
        #
        #                 # 提取摘要
        #                 abstract_elem = result.find('div', class_='c-abstract')
        #                 abstract = abstract_elem.get_text(strip=True) if abstract_elem else ''
        #
        #                 # 获取完整文章内容
        #                 full_content = self._get_article_content(link) if link else abstract
        #
        #                 # 使用模型分析内容
        #                 risk_analysis = self._analyze_risk(full_content, risk_model, risk_tokenizer)
        #                 sentiment_analysis = self._analyze_sentiment(full_content, sentiment_model, sentiment_tokenizer)
        #
        #                 results.append({
        #                     'title': title,
        #                     'content': full_content,
        #                     'link': link,
        #                     'source': '百度搜索',
        #                     'date': '未知',
        #                     'risk': risk_analysis,
        #                     'sentiment': sentiment_analysis
        #                 })
        #         except Exception as e:
        #             logger.warning(f"提取搜索结果时出错: {e}")
        #             continue
        #
        #     if not results:
        #         return "未找到相关新闻。"
        #
        #     output = "找到以下相关新闻：\n\n"
        #
        #     for i, result in enumerate(results, 1):
        #         output += f"{i}. {result['title']}\n"
        #         output += f"   来源: {result['source']}\n"
        #         if result['content']:
        #             content_preview = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
        #             output += f"   内容: {content_preview}\n"
        #         output += f"   风险分析: {result['risk']}\n"
        #         output += f"   情感分析: {result['sentiment']}\n"
        #         output += f"   链接: {result['link']}\n\n"
        #
        #     return output
        #
        # except Exception as e:
        #     logger.error(f"爬取新闻时出错: {e}")
        #     return f"爬取新闻时出错: {str(e)}"
    #
    # def _get_article_content(self, url: str) -> str:
    #     """
    #     获取文章的完整内容
    #
    #     Args:
    #         url: 文章链接
    #
    #     Returns:
    #         文章内容
    #     """
    #     try:
    #         import requests
    #         from bs4 import BeautifulSoup
    #
    #         headers = {
    #             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    #         }
    #
    #         response = requests.get(url, headers=headers, timeout=10)
    #         response.raise_for_status()
    #
    #         soup = BeautifulSoup(response.content, 'html.parser')
    #
    #         # 尝试多个内容选择器
    #         content_selectors = [
    #             'article p',
    #             '.article-content p',
    #             '.story-content p',
    #             '.post-content p',
    #             '.entry-content p',
    #             'p',
    #             '.content p'
    #         ]
    #
    #         content_parts = []
    #         for selector in content_selectors:
    #             paragraphs = soup.select(selector)
    #             if paragraphs:
    #                 for p in paragraphs:
    #                     text = p.get_text(strip=True)
    #                     if text and len(text) > 30:  # 只保留有意义的段落
    #                         content_parts.append(text)
    #                 break
    #
    #         return ' '.join(content_parts)
    #
    #     except Exception as e:
    #         logger.warning(f"获取文章内容时出错: {e}")
    #         return ""
    #
    def _load_risk_model(self):
        """加载风险模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            import torch
            
            risk_model_path = "/home/ruan/Finance/qwen_risk_model"
            base_model_name = "/home/ruan/Finance/Qwen3-1.7B"
            
            # 检查CUDA可用性
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"使用设备: {device}")
            
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.pad_token = tokenizer.eos_token

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 加载LoRA适配器
            risk_model = PeftModel.from_pretrained(base_model, risk_model_path)
            
            # 确保模型在正确的设备上
            if device == "cpu":
                risk_model = risk_model.to(device)
            
            logger.info("风险模型加载成功")
            return risk_model, tokenizer
            
        except Exception as e:
            logger.error(f"加载风险模型时出错: {e}")
            return None, None
    
    def _load_sentiment_model(self):
        """加载情感模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            import torch
            
            sentiment_model_path = "/home/ruan/Finance/qwen_sentiment_model"
            base_model_name = "/home/ruan/Finance/Qwen3-1.7B"
            
            # 检查CUDA可用性
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"使用设备: {device}")
            
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.pad_token = tokenizer.eos_token
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 加载LoRA适配器
            sentiment_model = PeftModel.from_pretrained(base_model, sentiment_model_path)
            
            # 确保模型在正确的设备上
            if device == "cpu":
                sentiment_model = sentiment_model.to(device)
            
            logger.info("情感模型加载成功")
            return sentiment_model, tokenizer
            
        except Exception as e:
            logger.error(f"加载情感模型时出错: {e}")
            return None, None
    
    def _analyze_risk(self, content: str, model, tokenizer) -> str:
        """使用风险模型分析内容"""
        try:
            if model is None or tokenizer is None:
                return "模型未加载"
            
            import torch
            
            # 获取模型所在设备
            device = next(model.parameters()).device
            
            # 构建风险评估提示词
            system_prompt = "Forget all your previous instructions. You are a financial expert specializing in risk assessment for stock recommendations. Based on a specific stock, provide a risk score from 1 to 5, where: 1 indicates very low risk, 2 indicates low risk, 3 indicates moderate risk (default if the news lacks any clear indication of risk), 4 indicates high risk, and 5 indicates very high risk. 1 summarized news will be passed in each time. Provide the score in the format shown below in the response from the assistant."
            
            user_content = f"News to Stock Symbol -- STOCK: {content}"
            
            prompt = f"""System: {system_prompt}

User: News to Stock Symbol -- AAPL: Apple (AAPL) increases 22%
Assistant: 3

User: News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30%
Assistant: 4

User: News to Stock Symbol -- AAPL: Apple (AAPL) announced iPhone 15
Assistant: 3

User: {user_content}
Assistant:"""
            
            # 编码输入并移动到正确的设备
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 生成预测
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取预测的风险分数
            assistant_response = generated_text.split("Assistant:")[-1].strip()
            
            # 尝试提取数字
            try:
                risk_score = int(assistant_response.split()[0])
                if 1 <= risk_score <= 5:
                    risk_map = {1: "极低风险", 2: "低风险", 3: "中等风险", 4: "高风险", 5: "极高风险"}
                    return f"{risk_score} ({risk_map[risk_score]})"
            except:
                pass
            
            return "无法分析风险"
            
        except Exception as e:
            logger.error(f"风险分析时出错: {e}")
            return f"风险分析失败: {str(e)}"
    
    def _analyze_sentiment(self, content: str, model, tokenizer) -> str:
        """使用情感模型分析内容"""
        try:
            if model is None or tokenizer is None:
                return "模型未加载"
            
            import torch
            
            # 获取模型所在设备
            device = next(model.parameters()).device
            
            # 构建情感分析提示词
            system_prompt = "Forget all your previous instructions. You are a financial expert with stock recommendation experience. Based on a specific stock, score for range from 1 to 5, where 1 is negative, 2 is somewhat negative, 3 is neutral, 4 is somewhat positive, 5 is positive. 1 summarized news will be passed in each time, you will give score in format as shown below in the response from assistant."
            
            user_content = f"News to Stock Symbol -- STOCK: {content}"
            
            prompt = f"""System: {system_prompt}

User: News to Stock Symbol -- AAPL: Apple (AAPL) increase 22%
Assistant: 5

User: News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30%
Assistant: 1

User: News to Stock Symbol -- AAPL: Apple (AAPL) announced iPhone 15
Assistant: 4

User: {user_content}
Assistant:"""
            
            # 编码输入并移动到正确的设备
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 生成预测
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # 解码输出
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取预测的情感分数
            assistant_response = generated_text.split("Assistant:")[-1].strip()
            
            # 尝试提取数字
            try:
                sentiment_score = int(assistant_response.split()[0])
                if 1 <= sentiment_score <= 5:
                    sentiment_map = {1: "负面", 2: "轻微负面", 3: "中性", 4: "正面", 5: "极正面"}
                    return f"{sentiment_score} ({sentiment_map[sentiment_score]})"
            except:
                pass
            
            return "无法分析情感"
            
        except Exception as e:
            logger.error(f"情感分析时出错: {e}")
            return f"情感分析失败: {str(e)}"