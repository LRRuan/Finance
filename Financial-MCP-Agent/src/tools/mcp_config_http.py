"""
MCP服务器配置模块 - 包含连接A股MCP服务器的配置信息
"""

SERVER_CONFIGS = {
    "a_share_mcp_v2": {
        "transport": "streamable_http",
        "url": "http://127.0.0.1:8080/mcp"
    }

}
