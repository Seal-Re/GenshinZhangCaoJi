# 原神 AI Agent 设计文档

**日期：** 2026-04-15  
**状态：** 已确认，待实现  
**项目目录：** `D:/AgentWorkStation/YuanShenAgent`

---

## 一、项目概述

构建一个用于原神（Genshin Impact）的综合 AI Agent，核心能力包括：

1. **游戏自动化**（核心）：自动开图、战斗、委托、活动，通过截图+视觉感知驱动
2. **视频学习开图**：输入B站攻略视频URL，自动提取路线并在游戏内复现
3. **数据查询**：Build分析、账号数据、深渊记录、游戏Wiki问答
4. **多界面交互**：Gradio Web UI + NoneBot2 QQ Bot 双入口
5. **Skill化任务系统**：任务定义为可复用Skill，支持定时调度与自然语言触发

**运行平台：** Windows（原神仅支持Windows/移动端）  
**VLM策略：** 多Provider，不锁定单一厂商（Claude / GPT-4o / Gemini，按场景路由）  
**用户API Key：** 自备

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Interface Layer                       │
│   Gradio Web UI (localhost:7860)   NoneBot2 QQ Bot      │
└──────────────────────────┬──────────────────────────────┘
                           │ AgentMessage 统一协议
┌──────────────────────────▼──────────────────────────────┐
│                 Agent Core (LangGraph)                   │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ SkillRouter │  │ TaskPlanner  │  │ APScheduler   │  │
│  │ (意图识别)   │  │ (任务拆解)   │  │ (定时任务)    │  │
│  └──────┬──────┘  └──────┬───────┘  └──────┬────────┘  │
└─────────┼────────────────┼─────────────────┼───────────┘
          └────────────────┼─────────────────┘
                           │ Skill 调用
┌──────────────────────────▼──────────────────────────────┐
│                    Skill Registry                        │
│  daily_commission │ explore_map │ domain_farm │ ...      │
└───────┬───────────────────────────────────┬─────────────┘
        │                                   │
┌───────▼────────┐               ┌──────────▼────────────┐
│  Data Layer    │               │  Game Control Layer   │
│ genshin.py     │               │  VisionClient         │
│ enka-py        │               │  CombatModule         │
│ genshin-db API │               │  NavigationModule     │
│ ambr.top API   │               │  DialogueModule       │
└────────────────┘               │  VideoRouteLearner    │
                                 └──────────────────────┘
```

---

## 三、目录结构

```
YuanShenAgent/
├── agent/
│   ├── core.py              # LangGraph 主图，Agent 入口
│   ├── planner.py           # 任务拆解节点
│   ├── router.py            # Skill 意图匹配
│   └── scheduler.py         # APScheduler 定时任务管理
│
├── skills/                  # Skill 定义（核心）
│   ├── _base.py             # BaseSkill 抽象类 + SkillContext + SkillResult
│   ├── daily_commission.py  # 每日委托
│   ├── explore_map.py       # 开图（含视频路线加载）
│   ├── domain_farm.py       # 刷副本/体力
│   ├── auto_dialogue.py     # 自动过剧情并总结
│   ├── daily_checkin.py     # 每日签到
│   ├── abyss_query.py       # 深渊记录查询
│   ├── build_analyzer.py    # 角色Build分析
│   ├── wiki_search.py       # 游戏Wiki问答
│   └── resin_check.py       # 树脂/便笺查询
│
├── vision/
│   ├── client.py            # VisionClient — 多VLM Provider 路由
│   ├── capture.py           # 截图 (mss)
│   ├── executor.py          # ActionExecutor — PyAutoGUI 键鼠封装
│   └── detector.py          # 游戏状态检测工具函数
│
├── scene/                   # Scene Module — 可复用场景处理器
│   ├── _base.py             # BaseSceneModule
│   ├── combat.py            # CombatModule
│   ├── navigation.py        # NavigationModule（含小地图解析）
│   ├── dialogue.py          # DialogueModule（OCR + LLM总结）
│   ├── settlement.py        # SettlementModule（VLM结算确认）
│   ├── chest.py             # ChestModule
│   └── ui.py                # UIModule（菜单/传送/背包等）
│
├── video/
│   ├── learner.py           # VideoRouteLearner — B站视频路线提取
│   ├── downloader.py        # yt-dlp 封装
│   ├── minimap_tracker.py   # 小地图帧追踪 + 路径点提取
│   └── annotator.py         # VLM关键帧语义标注
│
├── data/
│   ├── client.py            # DataClient — 统一数据访问入口
│   ├── genshin_client.py    # genshin.py 封装（账号/深渊/签到）
│   ├── enka_client.py       # enka-py 封装（角色展示柜）
│   └── game_db.py           # genshin-db API 封装（静态游戏数据）
│
├── interfaces/
│   ├── web_ui.py            # Gradio UI（对话+任务面板+截图流）
│   └── bot.py               # NoneBot2 QQ Bot（OneBot v11）
│
├── db/
│   └── routes.py            # 路线计划持久化（SQLite）
│
├── config/
│   ├── settings.yaml        # 全局配置（LLM/游戏/Bot）
│   └── schedule.yaml        # 定时任务配置
│
└── main.py                  # 启动入口（三服务并行）
```

---

## 四、Skill 系统

### 4.1 BaseSkill 接口

```python
class BaseSkill:
    name: str                    # 唯一标识
    description: str             # 用于 LLM 意图匹配的描述
    triggers: list[str]          # 自然语言触发词列表
    parameters: type[BaseModel]  # Pydantic 参数 Schema
    schedule: str | None         # cron 表达式，None = 仅手动触发

    async def run(self, ctx: SkillContext) -> SkillResult: ...
    async def on_interrupt(self): ...
    async def on_error(self, e: Exception): ...
```

### 4.2 SkillContext

```python
@dataclass
class SkillContext:
    trigger_source: Literal["chat", "scheduler", "agent"]
    params: dict
    vision: VisionClient
    executor: ActionExecutor
    data: DataClient
    reply: Callable[[AgentMessage], Awaitable[None]]
```

### 4.3 初期 Skill 清单

| Skill | 触发示例 | 核心依赖 | 实现难度 |
|---|---|---|---|
| `daily_commission` | "做委托" | NavigationModule + CombatModule + DialogueModule | 高 |
| `explore_map` | "开蒙德地图" | NavigationModule + VideoRouteLearner | 高（核心） |
| `domain_farm` | "清体力刷绝缘本" | CombatModule(CV) + SettlementModule | 中 |
| `auto_dialogue` | "自动过剧情并总结" | DialogueModule(OCR) + LLM总结 | 中 |
| `daily_checkin` | "签到" | genshin.py | 低 |
| `abyss_query` | "看我的深渊记录" | genshin.py | 低 |
| `build_analyzer` | "分析我的胡桃，uid=xxx" | enka-py + VLM | 低 |
| `wiki_search` | "万叶培养材料" | genshin-db + LLM | 低 |
| `resin_check` | "我还有多少树脂" | genshin.py | 低 |

---

## 五、游戏核心执行层

### 5.1 三层执行模型

```
Level 3: Skill Pipeline     ← 用户可见，自然语言触发
Level 2: Scene Module       ← 可复用场景处理器（跨Skill共享）
Level 1: Action Primitive   ← click / move / key_press / screenshot
```

### 5.2 Vision 双通道策略

| 通道 | 技术 | 延迟 | 适用场景 |
|---|---|---|---|
| CV Tier（快速） | OpenCV 模板匹配 / YOLO目标检测 | <50ms | 战斗、UI元素识别、小地图解析 |
| VLM Tier（理解） | Claude / GPT-4o / Gemini | 1-3s | 结算确认、剧情理解、阶段状态判断 |

```python
class VisionClient:
    async def fast_detect(self, frame, template) -> Detection:
        """OpenCV / YOLO，毫秒级"""

    async def understand(self, screenshot: bytes, prompt: str) -> str:
        """多VLM路由：按任务类型选最优Provider"""
        return await self._router.call(screenshot, prompt)
```

### 5.3 CombatModule

- **目标检测：** OpenCV 轮廓检测 + 血条颜色过滤（先用），YOLO微调（后优化）
- **战斗决策：** 预设队伍技能循环宏（Macro），支持按角色队伍配置
- **战斗状态机：** 执行循环 → 检测HP/CD → 闪避警告 → VLM确认结算
- **不使用VLM实时决策**：延迟不可接受

### 5.4 NavigationModule（开图核心）

```python
class MinimapAnalyzer:
    def get_unexplored_areas(self, minimap_img) -> list[Point]:
        """HSV过滤：已探索=彩色，未探索=灰色 → 灰色连通区域质心"""

    def get_player_position(self, minimap_img) -> Point:
        """玩家图标（白色箭头）模板匹配"""

    def detect_teleport_waypoints(self, minimap_img) -> list[Point]:
        """传送锚点图标模板匹配"""
```

**地形处理状态机：**
- 遇水 → 游泳
- 遇崖 → 攀爬
- 高空 → 滑翔
- 遇敌 → 触发 CombatModule
- 遇对话 → 触发 DialogueModule.skip()
- 遇宝箱 → 触发 ChestModule.open()

---

## 六、视频学习开图模块（VideoRouteLearner）

### 6.1 流程

```
B站URL → yt-dlp下载 → OpenCV提取小地图帧序列
       → 模板匹配追踪玩家图标 → 过滤传送跳变
       → 路径点序列 → VLM分析关键帧（宝箱/锚点/敌人/地名）
       → RoutePlan → SQLite存库 → ExploreMapSkill加载执行
```

### 6.2 使用方式

```
用户："学习这个开图视频：https://www.bilibili.com/video/BVxxxxxxx"

Agent（2-5分钟后）："✅ 已学习路线：层岩巨渊地下区域
  路线点：47个，宝箱：12个，锚点：3个
  输入「开图 层岩地下」开始执行"
```

### 6.3 可行性评估

| 环节 | 方案 | 可行性 |
|---|---|---|
| 下载B站视频 | yt-dlp | 高 |
| 小地图裁剪 | OpenCV 固定区域截取 | 高 |
| 玩家图标追踪 | 模板匹配白色箭头 | 高 |
| 传送跳变过滤 | 欧氏距离阈值判断 | 高 |
| 语义标注 | VLM分析关键帧 | 中 |
| 游戏内坐标对齐 | 小地图相对坐标映射（需一次校准） | 中 |

---

## 七、接口层

### 7.1 统一消息协议

```python
@dataclass
class AgentMessage:
    text: str
    images: list[bytes] = field(default_factory=list)
    skill_name: str | None = None
    progress: str | None = None   # 如 "3/4 委托完成"
```

### 7.2 Gradio Web UI

- 左侧：任务调度面板（已配置任务列表、启停控制、添加定时任务）
- 右侧：对话窗口（自然语言交互）
- 底部：实时截图流（技能执行快照，每步上报）

### 7.3 NoneBot2 QQ Bot

```
# 自然语言（LLM意图识别）
"帮我做委托"
"分析胡桃build，uid是12345678"

# 半结构化（直接触发，跳过意图识别）
/skill daily_commission
/skill explore_map region=蒙德
/skill build_analyze uid=12345678

# 调度管理
/schedule list
/schedule add domain_farm 22:00
/schedule pause daily_commission

# 视频学习
/learn https://www.bilibili.com/video/BVxxxxxxx
```

协议端：**Lagrange 或 NapCat**（OneBot v11 协议）

---

## 八、配置文件

```yaml
# config/settings.yaml
llm:
  default: claude-sonnet-4-6        # 对话/意图识别/剧情总结
  vision_fast: gemini-2.0-flash     # VLM视觉（低延迟优先）
  vision_fallback: gpt-4o           # 备用VLM

genshin:
  uid: ""                           # 玩家UID
  cookie_path: .secrets/cookie.txt  # genshin.py HoYoLAB认证

game:
  window_title: "原神"
  screenshot_interval_ms: 500       # 截图间隔（战斗时缩短）
  minimap_region: [1820, 30, 95, 95] # 小地图裁剪区域（需按分辨率校准）

bot:
  platform: qq
  onebot_host: 127.0.0.1
  onebot_port: 6700
  superusers: []                    # 管理员QQ号列表
```

---

## 九、核心依赖清单

```bash
# Agent 框架
pip install langchain-anthropic langgraph apscheduler

# 原神数据 API
pip install genshin enka ambr

# 游戏视觉与控制
pip install mss pyautogui opencv-python
pip install ultralytics          # YOLOv8（后期目标检测优化）
pip install paddleocr             # OCR（对话框文字提取）

# 视频处理
pip install yt-dlp

# 接口层
pip install gradio
pip install nonebot2 nonebot-adapter-onebot

# 工具
pip install pydantic sqlalchemy aiofiles
```

---

## 十、开发优先级

**Phase 1（核心可用）**
1. 项目骨架 + BaseSkill + SkillContext
2. VisionClient 双通道（OpenCV + 一个VLM Provider）
3. ActionExecutor（PyAutoGUI封装）
4. NavigationModule + MinimapAnalyzer（开图基础）
5. CombatModule（OpenCV检测 + 基础Macro）
6. ExploreMapSkill（手动触发开图）

**Phase 2（自动化闭环）**
7. DialogueModule + auto_dialogue Skill
8. DailyCommissionSkill（委托自动化）
9. APScheduler 定时任务
10. Gradio Web UI

**Phase 3（学习与扩展）**
11. VideoRouteLearner（B站视频学路线）
12. NoneBot2 QQ Bot
13. 数据层 Skills（签到/深渊/Build分析）
14. DomainFarmSkill（刷副本）
