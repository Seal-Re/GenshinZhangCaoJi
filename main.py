# main.py
"""
CLI entry point for YuanShen Agent — Phase 1.

Usage:
    python main.py "开图"
    python main.py "探索蒙德地图"
    python main.py --list-skills
"""
from __future__ import annotations
import argparse
import asyncio
import os
import sys


async def run_cli(user_input: str) -> None:
    from agent.core import build_agent
    from skills._base import AgentMessage

    print(f"[Agent] 收到指令: {user_input}")
    agent, router = build_agent()

    result = await agent.ainvoke({
        "user_input": user_input,
        "skill_name": None,
        "skill_params": {},
        "result_message": "",
        "reply_messages": [],
    })

    for msg in result.get("reply_messages", []):
        prefix = f"[{msg.skill_name}]" if msg.skill_name else "[Agent]"
        if msg.progress:
            print(f"{prefix} {msg.text} ({msg.progress})")
        else:
            print(f"{prefix} {msg.text}")

    print(f"[完成] {result['result_message']}")


def list_skills() -> None:
    from skills.explore_map import ExploreMapSkill
    skills = [ExploreMapSkill()]
    print("已注册技能:")
    for s in skills:
        triggers = " / ".join(s.triggers[:3])
        print(f"  {s.name}: {s.description[:50]}...")
        print(f"    触发词: {triggers}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YuanShen Agent CLI")
    parser.add_argument("input", nargs="?", help="Natural language command")
    parser.add_argument("--list-skills", action="store_true")
    args = parser.parse_args()

    if args.list_skills:
        list_skills()
        sys.exit(0)

    if not args.input:
        parser.print_help()
        sys.exit(1)

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("错误: 请设置 ANTHROPIC_API_KEY 环境变量")
        sys.exit(1)

    asyncio.run(run_cli(args.input))
