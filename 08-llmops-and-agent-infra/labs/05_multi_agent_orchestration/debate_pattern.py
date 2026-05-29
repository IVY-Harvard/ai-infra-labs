"""
Lab 05: Debate Pattern（辩论模式）
两个 Agent 就某个议题辩论，裁判总结最终结论
"""
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-72b-instruct")


def get_llm(temperature=0.7):
    return ChatOpenAI(
        base_url=LLM_BASE_URL, model=LLM_MODEL,
        api_key="not-needed", temperature=temperature,
    )


class DebateAgent:
    """辩论 Agent"""

    def __init__(self, name: str, stance: str, system_prompt: str):
        self.name = name
        self.stance = stance
        self.system_prompt = system_prompt
        self.llm = get_llm(temperature=0.7)

    def argue(self, topic: str, opponent_args: str = "",
              round_num: int = 1) -> str:
        prompt = ChatPromptTemplate.from_template(
            """{system_prompt}

辩论主题：{topic}
你的立场：{stance}
当前轮次：第 {round_num} 轮

{opponent_section}

请从你的立场出发进行论证。要求：
1. 提出有力的论点和证据
2. 如果对方有论点，针对性反驳
3. 回复控制在 150 字以内
4. 保持专业和理性

你的论述："""
        )

        opponent_section = (
            f"对方论点：\n{opponent_args}" if opponent_args
            else "（第一轮，暂无对方论点）"
        )

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "system_prompt": self.system_prompt,
            "topic": topic,
            "stance": self.stance,
            "round_num": round_num,
            "opponent_section": opponent_section,
        })


class JudgeAgent:
    """裁判 Agent"""

    def __init__(self):
        self.llm = get_llm(temperature=0)

    def judge(self, topic: str, debate_history: list[dict]) -> str:
        """裁决辩论结果"""
        history_text = "\n\n".join([
            f"[Round {d['round']}]\n"
            f"  正方({d['pro_name']}): {d['pro_arg']}\n"
            f"  反方({d['con_name']}): {d['con_arg']}"
            for d in debate_history
        ])

        prompt = ChatPromptTemplate.from_template(
            """你是一个公正的技术评审。请根据双方辩论内容做出裁决。

辩论主题：{topic}

辩论记录：
{history}

请给出裁决：
1. 双方论点总结
2. 各自的优势和不足
3. 最终结论和建议（综合双方观点的平衡结论）
4. 给出评分（正方 X 分 / 反方 Y 分，满分 10 分）

裁决："""
        )

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "topic": topic,
            "history": history_text,
        })


class DebateOrchestrator:
    """辩论编排器"""

    def __init__(self, topic: str, pro_stance: str, con_stance: str):
        self.topic = topic
        self.pro = DebateAgent(
            name="正方",
            stance=pro_stance,
            system_prompt=f"你是辩论正方，坚定支持：{pro_stance}。用数据和案例支持你的观点。",
        )
        self.con = DebateAgent(
            name="反方",
            stance=con_stance,
            system_prompt=f"你是辩论反方，坚定支持：{con_stance}。用数据和案例支持你的观点。",
        )
        self.judge = JudgeAgent()

    def run(self, num_rounds: int = 3) -> str:
        """运行辩论"""
        print(f"\n{'='*60}")
        print(f"辩论主题：{self.topic}")
        print(f"正方立场：{self.pro.stance}")
        print(f"反方立场：{self.con.stance}")
        print(f"{'='*60}")

        debate_history = []
        pro_last_arg = ""
        con_last_arg = ""

        for round_num in range(1, num_rounds + 1):
            print(f"\n--- Round {round_num} ---")

            # 正方发言
            pro_arg = self.pro.argue(
                self.topic, con_last_arg, round_num
            )
            print(f"\n  [正方]: {pro_arg}")

            # 反方发言
            con_arg = self.con.argue(
                self.topic, pro_arg, round_num
            )
            print(f"\n  [反方]: {con_arg}")

            debate_history.append({
                "round": round_num,
                "pro_name": self.pro.name,
                "con_name": self.con.name,
                "pro_arg": pro_arg,
                "con_arg": con_arg,
            })

            pro_last_arg = pro_arg
            con_last_arg = con_arg

        # 裁判裁决
        print(f"\n{'='*60}")
        print(f"裁判裁决")
        print(f"{'='*60}")

        verdict = self.judge.judge(self.topic, debate_history)
        print(f"\n{verdict}")

        return verdict


def main():
    # 辩论 1：向量数据库选型
    debate1 = DebateOrchestrator(
        topic="企业级 RAG 系统应该选择 Milvus 还是 Qdrant？",
        pro_stance="应该选择 Milvus，因为它是分布式架构，适合大规模数据",
        con_stance="应该选择 Qdrant，因为它性能更好且运维更简单",
    )
    debate1.run(num_rounds=3)

    # 辩论 2：Agent 框架
    debate2 = DebateOrchestrator(
        topic="生产环境 Agent 开发应该用 LangGraph 还是 AutoGen？",
        pro_stance="应该用 LangGraph，状态管理和控制流更精确",
        con_stance="应该用 AutoGen，对话式交互更灵活、开发更快",
    )
    debate2.run(num_rounds=2)


if __name__ == "__main__":
    main()
