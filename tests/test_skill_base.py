import pytest
from unittest.mock import AsyncMock, MagicMock
from skills._base import BaseSkill, SkillContext, SkillResult, AgentMessage


class ConcreteSkill(BaseSkill):
    name = "test_skill"
    description = "A test skill"
    triggers = ["test", "testing"]
    schedule = None

    async def run(self, ctx: SkillContext) -> SkillResult:
        await ctx.reply(AgentMessage(text="hello"))
        return SkillResult(success=True, message="done")


@pytest.fixture
def mock_ctx():
    return SkillContext(
        trigger_source="chat",
        params={},
        capture=MagicMock(),
        vision=MagicMock(),
        executor=MagicMock(),
        data=MagicMock(),
        reply=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_skill_run_returns_result(mock_ctx):
    skill = ConcreteSkill()
    result = await skill.run(mock_ctx)
    assert result.success is True
    assert result.message == "done"


@pytest.mark.asyncio
async def test_skill_run_calls_reply(mock_ctx):
    skill = ConcreteSkill()
    await skill.run(mock_ctx)
    mock_ctx.reply.assert_called_once()
    call_arg = mock_ctx.reply.call_args[0][0]
    assert isinstance(call_arg, AgentMessage)
    assert call_arg.text == "hello"


@pytest.mark.asyncio
async def test_skill_on_error_returns_failure(mock_ctx):
    skill = ConcreteSkill()
    result = await skill.on_error(ValueError("oops"))
    assert result.success is False
    assert "oops" in result.message


def test_agent_message_defaults():
    msg = AgentMessage(text="hi")
    assert msg.images == []
    assert msg.skill_name is None
    assert msg.progress is None
