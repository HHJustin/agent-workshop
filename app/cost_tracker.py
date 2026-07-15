"""
成本追踪 — Token 计数 + 费用估算 + 按轮次/会话汇总

DashScope 千问定价（元/百万 token）：
    qwen-max:   输入 0.02  输出 0.06
    qwen-turbo: 输入 0.008 输出 0.024

Author: 程响
"""

from app.logger import logger

# 千问模型定价（元/百万 token）
PRICING = {
    "qwen-max":   {"input": 0.02,  "output": 0.06},
    "qwen-turbo": {"input": 0.008, "output": 0.024},
    "default":    {"input": 0.01,  "output": 0.03},
}


class CostTracker:
    """单轮对话的成本追踪器"""

    def __init__(self, model: str = ""):
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0

    def record(self, usage: dict):
        """记录一次 LLM 调用的 token 消耗"""
        self.llm_calls += 1
        self.input_tokens += usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_rmb(self) -> float:
        """估算费用（人民币元）"""
        price = PRICING.get(self.model, PRICING["default"])
        cost = (self.input_tokens / 1_000_000) * price["input"] + \
               (self.output_tokens / 1_000_000) * price["output"]
        return round(cost, 6)

    @property
    def summary(self) -> str:
        return (
            f"LLM调用{self.llm_calls}次, "
            f"输入{self.input_tokens}T + 输出{self.output_tokens}T = {self.total_tokens}T, "
            f"预估费用 ¥{self.estimated_cost_rmb:.4f}"
        )


# 全局会话级汇总
_session_costs: dict[str, list[CostTracker]] = {}


def track_turn(session_id: str, cost: CostTracker):
    """记录一轮对话的成本到会话汇总"""
    if session_id not in _session_costs:
        _session_costs[session_id] = []
    _session_costs[session_id].append(cost)


def get_session_cost(session_id: str) -> dict:
    """获取会话累计成本"""
    turns = _session_costs.get(session_id, [])
    total_input = sum(c.input_tokens for c in turns)
    total_output = sum(c.output_tokens for c in turns)
    total_cost = sum(c.estimated_cost_rmb for c in turns)
    return {
        "turns": len(turns),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "estimated_cost_rmb": round(total_cost, 4),
    }
