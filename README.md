# Deep Research Framework

基于 OpenAI Deep Research API 的异步研究报告生成框架。提交研究任务后，系统自动调用业务 API 采集数据、生成带交互式图表的专业分析报告。

## 架构概览

```
Client → FastAPI → Background Task
                      ├── Project Config (YAML)
                      ├── Data Preparation (API prefetch + docs)
                      ├── Prompt Builder (research prompt + user profile)
                      ├── Research Engine (OpenAI Deep Research)
                      ├── Output Renderer (HTML/PDF/Markdown)
                      └── User Memory (profile generation & personalization)
```

**核心流程：**

1. 客户端提交研究任务（项目、查询、API Key）
2. 系统加载项目配置，预取业务数据和 API 文档
3. 构建研究 prompt，调用 OpenAI Deep Research API
4. AI 在研究过程中可动态调用业务 API 获取更多数据
5. 渲染输出为 HTML（含 ECharts 图表）、PDF 或 Markdown
6. 记录用户交互历史，LLM 自动生成用户画像，后续研究更懂你

## 用户记忆系统

框架内置基于 LLM 的用户记忆系统，随着使用自动学习用户偏好，让每次研究报告更贴合你的需求。

**工作原理：**

- 以 `sha256(api_key)` 作为匿名用户标识，不存储原始 Key
- 每次研究完成后，自动记录交互历史（查询 + 报告摘要）
- LLM 分析历史交互，自主发现有价值的用户特征维度（不限于预设维度）
- 用户画像注入研究 Prompt，让 AI 更了解你的关注点和偏好

**自动发现的维度示例：**

用户画像由 LLM 根据实际交互自由生成，可能包括但不限于：关注的行业/市场/产品、分析习惯、专业背景、偏好的报告风格等。

**数据存储：** SQLite（WAL 模式），与任务数据共享同一数据库文件，零额外运维成本。

## 快速开始

### 环境变量

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-xxx          # OpenAI API Key
API_TOKEN=your-bearer-token     # 服务鉴权 Token
PORT=8000                       # 服务端口（可选）
```

### Docker 部署（推荐）

```bash
PORT=8000 docker compose up -d --build
```

### 本地开发

```bash
# Python 3.12+
pip install -e ".[dev]"
uvicorn src.main:app --reload
```

## API 接口

### 提交研究任务

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_TOKEN" \
  -d '{
    "project": "admapix",
    "query": "分析某游戏近30天的广告投放策略",
    "api_key": "user-business-api-key"
  }'
```

返回 `task_id`，任务在后台异步执行。

### 查询任务状态

```bash
curl http://localhost:8000/research/{task_id} \
  -H "Authorization: Bearer $API_TOKEN"
```

返回值包含 `status`（pending / processing / completed / failed）、进度信息、以及完成后的报告文件链接。

### 其他接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/projects` | 项目列表 |
| POST | `/research/{task_id}/cancel` | 取消任务 |
| GET | `/files/{task_id}/{filename}` | 下载报告文件 |
| GET | `/incidents` | API 故障记录 |

## 项目配置

每个研究项目是 `projects/` 下的一个目录，包含：

```
projects/admapix/
├── config.yaml          # 项目配置（API、模型、系统指令）
├── output_prefs.yaml    # 输出偏好（语言、格式提示）
└── api_docs/            # 业务 API 文档（Markdown）
    ├── api-creative.md
    ├── api-distribution.md
    └── ...
```

### config.yaml 示例

```yaml
name: admapix
description: "广告素材情报分析平台"

apis:
  - name: admapix_main
    base_url: https://api.example.com
    auth:
      type: header           # header 或 bearer
      header_name: X-API-Key
      token_env: ADMAPIX_API_KEY  # 环境变量名（fallback）
    docs_files:
      - api_docs/api-creative.md
    prefetch:                # 预取数据（注入到研究上下文）
      - endpoint: /api/data/ranking
        method: POST
        body: { rank_type: promotion, page_size: 20 }

model: gpt-5.4              # 使用的 OpenAI 模型

system_instructions: |
  你是一名资深的广告投放分析师...
```

### API Key 传递

业务 API Key 由调用方在请求中通过 `api_key` 字段传入，框架不在服务端存储用户的业务 Key。如果请求未携带 `api_key`，返回 422 错误。

## 技术栈

- **Python 3.12+** / FastAPI / Uvicorn
- **OpenAI API** (Deep Research)
- **SQLite** (aiosqlite) — 任务状态 & 用户记忆存储
- **httpx** — 异步 HTTP 请求
- **Docker** — 部署 & 沙箱渲染

## 项目结构

```
src/
├── main.py                  # FastAPI 应用入口
├── config.py                # 配置管理 (pydantic-settings)
├── auth.py                  # Bearer Token 鉴权
├── api/
│   ├── models.py            # 请求/响应模型
│   └── routes.py            # API 路由 & 研究任务调度
├── engine/
│   ├── project_loader.py    # 项目配置加载
│   ├── prompt_builder.py    # 研究 Prompt 构建
│   ├── data_preparation.py  # 数据预取 & API 调用
│   └── research.py          # Deep Research 引擎
├── memory/
│   ├── store.py             # 用户记忆存储 (SQLite)
│   └── updater.py           # LLM 驱动的用户画像生成
├── output/
│   └── renderer.py          # 报告渲染 (HTML/PDF/MD)
└── task/
    └── manager.py           # 任务生命周期管理 (SQLite)
```

## License

MIT
