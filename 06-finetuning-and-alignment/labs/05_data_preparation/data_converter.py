"""
数据格式转换工具：在 Alpaca、ShareGPT、OpenAI Messages 格式之间互转

支持的格式:
- alpaca:    {"instruction": ..., "input": ..., "output": ...}
- sharegpt:  {"conversations": [{"from": "human/gpt", "value": ...}]}
- messages:  {"messages": [{"role": "user/assistant", "content": ...}]}

用法:
    python data_converter.py --input data.json --from alpaca --to sharegpt
    python data_converter.py --input data.jsonl --from sharegpt --to messages --output out.jsonl
"""

import argparse
import json
import os
from typing import List, Dict, Any


class DataConverter:
    """多格式数据转换器"""

    @staticmethod
    def load_data(file_path: str) -> List[Dict]:
        """加载 JSON 或 JSONL 格式数据"""
        data = []
        if file_path.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    data = loaded
                else:
                    data = [loaded]
        return data

    @staticmethod
    def save_data(data: List[Dict], file_path: str):
        """保存为 JSON 或 JSONL"""
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        if file_path.endswith(".jsonl"):
            with open(file_path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(data)} 条数据到 {file_path}")

    # ========================
    # 转为统一中间格式 (messages)
    # ========================
    @staticmethod
    def alpaca_to_messages(item: Dict) -> Dict:
        """Alpaca → Messages"""
        messages = []
        if item.get("system"):
            messages.append({"role": "system", "content": item["system"]})

        user_content = item.get("instruction", "")
        if item.get("input"):
            user_content += f"\n{item['input']}"

        messages.append({"role": "user", "content": user_content.strip()})
        messages.append({"role": "assistant", "content": item.get("output", "")})

        result = {"messages": messages}
        # 保留额外元数据
        for key in item:
            if key not in ("instruction", "input", "output", "system"):
                result[key] = item[key]
        return result

    @staticmethod
    def sharegpt_to_messages(item: Dict) -> Dict:
        """ShareGPT → Messages"""
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        messages = []

        conversations = item.get("conversations", [])
        for conv in conversations:
            role = role_map.get(conv.get("from", ""), conv.get("from", ""))
            content = conv.get("value", "")
            messages.append({"role": role, "content": content})

        result = {"messages": messages}
        for key in item:
            if key != "conversations":
                result[key] = item[key]
        return result

    # ========================
    # 从中间格式转出
    # ========================
    @staticmethod
    def messages_to_alpaca(item: Dict) -> Dict:
        """Messages → Alpaca"""
        messages = item.get("messages", [])

        system = ""
        instruction = ""
        output = ""

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            elif msg["role"] == "user":
                instruction = msg["content"]
            elif msg["role"] == "assistant":
                output = msg["content"]

        result = {"instruction": instruction, "input": "", "output": output}
        if system:
            result["system"] = system
        return result

    @staticmethod
    def messages_to_sharegpt(item: Dict) -> Dict:
        """Messages → ShareGPT"""
        role_map = {"user": "human", "assistant": "gpt", "system": "system"}
        conversations = []

        for msg in item.get("messages", []):
            conversations.append({
                "from": role_map.get(msg["role"], msg["role"]),
                "value": msg["content"],
            })

        return {"conversations": conversations}

    # ========================
    # 主转换接口
    # ========================
    def convert(self, data: List[Dict], from_format: str, to_format: str) -> List[Dict]:
        """执行格式转换"""
        # 先转为 messages 中间格式
        if from_format == "alpaca":
            intermediate = [self.alpaca_to_messages(item) for item in data]
        elif from_format == "sharegpt":
            intermediate = [self.sharegpt_to_messages(item) for item in data]
        elif from_format == "messages":
            intermediate = data
        else:
            raise ValueError(f"不支持的源格式: {from_format}")

        # 再从 messages 转为目标格式
        if to_format == "messages":
            return intermediate
        elif to_format == "alpaca":
            return [self.messages_to_alpaca(item) for item in intermediate]
        elif to_format == "sharegpt":
            return [self.messages_to_sharegpt(item) for item in intermediate]
        else:
            raise ValueError(f"不支持的目标格式: {to_format}")


def main():
    parser = argparse.ArgumentParser(description="数据格式转换工具")
    parser.add_argument("--input", required=True, help="输入文件路径")
    parser.add_argument("--from", dest="from_format", required=True,
                       choices=["alpaca", "sharegpt", "messages"])
    parser.add_argument("--to", dest="to_format", required=True,
                       choices=["alpaca", "sharegpt", "messages"])
    parser.add_argument("--output", default=None, help="输出文件路径")
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_{args.to_format}{ext}"

    converter = DataConverter()

    # 加载
    print(f"加载数据: {args.input}")
    data = converter.load_data(args.input)
    print(f"读取 {len(data)} 条数据 (格式: {args.from_format})")

    # 转换
    converted = converter.convert(data, args.from_format, args.to_format)
    print(f"转换完成: {args.from_format} → {args.to_format}")

    # 保存
    converter.save_data(converted, args.output)

    # 预览
    print("\n转换后样本预览:")
    print(json.dumps(converted[0], ensure_ascii=False, indent=2)[:500])


if __name__ == "__main__":
    main()
