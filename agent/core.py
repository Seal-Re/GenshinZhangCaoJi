from __future__ import annotations
import os
from typing import TypedDict

from langgraph.graph import StateGraph, END

from agent.router import SkillRouter
from config.loader import load_config
from data.client import DataClient
from skills._base import AgentMessage, SkillContext
from skills.explore_map import ExploreMapSkill
from vision.capture import ScreenCapture
from vision.client import VisionClient
from vision.executor import ActionExecutor
from vision.providers import ClaudeVLMProvider, OpenAIVLMProvider


class AgentState(TypedDict):
    user_input: str
    skill_name: str | None
    skill_params: dict
    result_message: str
    reply_messages: list[AgentMessage]


def _build_vlm_provider(cfg):
    """Select VLM provider based on config."""
    if cfg.llm.provider == "openai":
        return OpenAIVLMProvider(
            api_key=cfg.openai.api_key,
            model=cfg.openai.model_id,
            base_url=cfg.openai.base_url or None,
        )
    # Default: Claude
    return ClaudeVLMProvider(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model=cfg.llm.default,
    )


def build_agent() -> tuple[object, SkillRouter]:
    cfg = load_config()

    # Wire up infrastructure
    capture = ScreenCapture()
    executor = ActionExecutor()
    vlm = _build_vlm_provider(cfg)
    vision = VisionClient(vlm_provider=vlm)
    data = DataClient()

    # Register skills
    skills = [ExploreMapSkill()]
    router = SkillRouter(skills)

    async def _route_node(state: AgentState) -> AgentState:
        skill = router.match(state["user_input"])
        if skill is None:
            return {**state, "skill_name": None,
                    "result_message": "未找到匹配的技能，请尝试: " +
                    ", ".join(s.triggers[0] for s in router.all_skills)}
        return {**state, "skill_name": skill.name}

    async def _execute_node(state: AgentState) -> AgentState:
        if state["skill_name"] is None:
            return state
        skill = router._skills[state["skill_name"]]
        replies: list[AgentMessage] = []

        async def collect_reply(msg: AgentMessage) -> None:
            replies.append(msg)

        ctx = SkillContext(
            trigger_source="chat",
            params=state["skill_params"],
            capture=capture,
            vision=vision,
            executor=executor,
            data=data,
            reply=collect_reply,
        )
        result = await skill.run(ctx)
        return {**state, "result_message": result.message,
                "reply_messages": replies}

    graph = StateGraph(AgentState)
    graph.add_node("route", _route_node)
    graph.add_node("execute", _execute_node)
    graph.set_entry_point("route")
    graph.add_edge("route", "execute")
    graph.add_edge("execute", END)

    return graph.compile(), router
