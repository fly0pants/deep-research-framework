# Deep Research Framework - Design Spec

## 1. Overview

一个独立部署的深度研究服务，基于 OpenAI Deep Research API，为 OpenClaw skill（如 admapix）提供复杂问题的深度研究和报告生成能力。

**核心能力：**
- 接收研究任务，调用 OpenAI Deep Research API 进行多步骤自主研究
- 通过内置 MCP Server 将项目 API 数据源暴露给 Deep Research 模型，实现业务数据 + 网络搜索的自动整合
- 模型自主决策最佳输出格式（PDF / HTML / 交互式图表等），生成渲染代码并执行
- 通过 HTTP API 提供服务，Docker 部署

## 2. Architecture

```
调用方 (OpenClaw skill / curl)
    ↓ HTTP POST /research
┌──────────────────────────────────────────────┐
│  Deep Research Service (FastAPI)             │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │ Request Handler                        │  │
│  │ - 验证请求 & API Token                 │  │
│  │ - 加载项目配置 & API 文档               │  │
│  │ - 创建异步研究任务 (UUID v4)            │  │
│  └──────────────┬─────────────────────────┘  │
│                 ↓                            │
│  ┌────────────────────────────────────────┐  │
│  │ Data Preparation Layer                 │  │
│  │ - 根据项目配置预调用项目 API            │  │
│  │ - 将业务数据写入临时知识文件             │  │
│  │ - 构建研究 Prompt                      │  │
│  └──────────────┬─────────────────────────┘  │
│                 ↓                            │
│  ┌────────────────────────────────────────┐  │
│  │ Research Engine                        │  │
│  │ - 调用 OpenAI Deep Research API        │  │
│  │   (background=True)                    │  │
│  │ - tools: [web_search, code_interpreter │  │
│  │           file_search (项目数据)]       │  │
│  │ - 轮询研究进度                          │  │
│  │ - 解析中间步骤 & 最终结果               │  │
│  └──────────────┬─────────────────────────┘  │
│                 ↓                            │
│  ┌────────────────────────────────────────┐  │
│  │ Output Engine                          │  │
│  │ - 模型决策输出格式                      │  │
│  │ - 生成渲染代码                          │  │
│  │ - Docker 隔离容器执行代码               │  │
│  │ - 产出最终文件                          │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
        ↓
  /output/{task_id}/  (PDF/HTML/图表等)
```

## 3. Project Configuration

每个项目一个独立目录，包含该项目相关的所有配置：

```
projects/
├── admapix/
│   ├── config.yaml              # 项目主配置
│   ├── api_docs/                # API 文档
│   │   ├── admapix_api.yaml     # OpenAPI spec
│   │   └── README.md            # API 使用说明（可选）
│   └── output_prefs.yaml        # 输出偏好（可选）
├── another_project/
│   ├── config.yaml
│   └── api_docs/
│       └── ...
```

### config.yaml 示例

```yaml
name: admapix
description: "广告素材管理与分析平台"

# API 数据源
apis:
  - name: admapix_main
    base_url: https://api.admapix.com/v1
    auth:
      type: bearer
      token_env: ADMAPIX_API_KEY       # 引用环境变量
    docs_file: api_docs/admapix_api.yaml
    # 预拉取配置：研究前自动调用这些端点获取基础数据
    prefetch:
      - endpoint: /campaigns/summary
        params: { days: 30 }
      - endpoint: /creatives/stats
        params: { days: 30 }

# 研究模型配置（可选，覆盖全局默认）
model: o3-deep-research

# 研究指令（可选，附加到 prompt）
system_instructions: |
  你是广告数据分析专家。分析时关注 ROI、CTR、CVR 等核心指标。
```

### output_prefs.yaml 示例

```yaml
# 可选，如果不提供则完全由模型自主决策
preferred_language: zh-CN
branding:
  logo_url: https://example.com/logo.png
  primary_color: "#1a73e8"
hints:
  - "数据密集型结果优先使用交互式图表"
  - "对比分析使用表格"
```

## 4. API Design

### 4.1 提交研究任务

```
POST /research
Authorization: Bearer {API_TOKEN}
Content-Type: application/json

{
  "project": "admapix",                    // 必填：项目标识
  "query": "分析近30天广告素材效果趋势",      // 必填：研究主题
  "context": "重点关注视频类素材",            // 可选：补充上下文
  "callback_url": "https://..."            // 可选：完成后 webhook 回调
}

Response 202:
{
  "task_id": "dr_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending",
  "created_at": "2026-03-18T10:00:00Z"
}
```

### 4.2 查询任务状态 & 获取结果（合并端点）

```
GET /research/{task_id}
Authorization: Bearer {API_TOKEN}

// 进行中：
Response 200:
{
  "task_id": "dr_a1b2c3d4-...",
  "status": "processing",
  "progress": {
    "stage": "researching",                // preparing | researching | rendering
    "message": "正在搜索广告效果数据..."
  },
  "created_at": "2026-03-18T10:00:00Z",
  "updated_at": "2026-03-18T10:01:30Z"
}

// 完成后（同一端点，status=completed 时包含完整结果）：
Response 200:
{
  "task_id": "dr_a1b2c3d4-...",
  "status": "completed",
  "output": {
    "format": "html",
    "files": [
      {
        "name": "report.html",
        "url": "/files/dr_a1b2c3d4-.../report.html",
        "type": "text/html",
        "size": 45230
      }
    ],
    "summary": "近30天广告素材效果整体呈上升趋势...",
    "sources": [
      {"type": "api", "name": "admapix_main", "calls": 3},
      {"type": "web", "url": "https://...", "title": "..."}
    ]
  },
  "usage": {
    "model": "o3-deep-research",
    "total_tokens": 15000,
    "research_time_seconds": 120
  }
}
```

### 4.3 取消任务

```
POST /research/{task_id}/cancel
Authorization: Bearer {API_TOKEN}

Response 200:
{
  "task_id": "dr_a1b2c3d4-...",
  "status": "cancelled"
}
```

### 4.4 获取文件

```
GET /files/{task_id}/{filename}
Authorization: Bearer {API_TOKEN}

Response: 文件二进制内容
```

### 4.5 健康检查

```
GET /health

Response 200:
{
  "status": "ok",
  "openai_api": "connected",
  "active_tasks": 3
}
```

### 4.6 列出项目

```
GET /projects
Authorization: Bearer {API_TOKEN}

Response 200:
{
  "projects": [
    {"name": "admapix", "description": "广告素材管理与分析平台", "apis": 1},
    {"name": "another_project", "description": "...", "apis": 2}
  ]
}
```

### 4.7 错误响应格式

所有错误返回统一格式：

```json
{
  "error": {
    "code": "project_not_found",
    "message": "Project 'xyz' does not exist",
    "details": {"available_projects": ["admapix", "another_project"]}
  }
}
```

## 5. Core Flow

```
1. 接收请求
   ├── 验证 API Token
   ├── 校验 project 存在
   ├── 检查并发限制（信号量）
   └── 创建任务记录（UUID v4），返回 task_id

2. 数据准备（Data Preparation）
   ├── 读取 config.yaml + API 文档 + 输出偏好
   ├── 执行 prefetch 配置的 API 端点，获取业务数据
   ├── 将业务数据 + API 文档写入临时文件集
   └── 创建 OpenAI Vector Store，上传文件集

3. 构建研究请求
   ├── 系统指令：研究者角色 + 项目 system_instructions + 输出偏好
   ├── 用户问题 + context
   └── tools 配置：
       ├── web_search_preview（网络搜索）
       ├── code_interpreter（代码执行 & 分析）
       └── file_search（检索 Vector Store 中的业务数据）

4. 调用 OpenAI Deep Research API
   ├── model: config 中指定或全局默认
   ├── background: True（异步执行）
   ├── 轮询等待完成（指数退避，5s → 10s → 20s → 30s max）
   └── 模型自主决定何时用 API 数据、何时搜索网络

5. 输出渲染
   ├── 解析模型最终输出
   ├── 模型已通过 code_interpreter 生成了部分可视化
   ├── 如需额外渲染（PDF转换等），在 Docker 隔离容器中执行
   │   ├── --network=none（禁止网络）
   │   ├── 只挂载 /output/{task_id}/ 目录
   │   ├── 内存限制 512MB，CPU 限制 1 核
   │   └── 执行超时 60 秒
   ├── 渲染失败 → 降级为 Markdown 输出
   └── 保存到 /output/{task_id}/

6. 完成通知
   ├── 更新任务状态为 completed
   ├── 清理临时 Vector Store
   └── 如有 callback_url（仅 HTTPS，过滤私有 IP），POST 通知
```

## 6. Data Integration Strategy

**核心思路：预拉取 + file_search，而非自定义 function calling。**

Deep Research API 不支持传统 function calling，但支持 `file_search`（基于 Vector Store）。因此采用以下策略：

### 阶段一：数据预拉取（框架执行）

在调用 Deep Research 前，框架根据项目配置主动获取业务数据：

1. 执行 `config.yaml` 中 `prefetch` 配置的 API 端点
2. 将返回的数据格式化为结构化文档（JSON/Markdown）
3. 连同 API 文档一起上传到 OpenAI Vector Store
4. 在 Deep Research 调用中通过 `file_search` 工具引用

### 阶段二：Deep Research 自主研究

模型在研究过程中可以：
- 通过 `file_search` 检索预拉取的业务数据
- 通过 `web_search_preview` 搜索网络补充信息
- 通过 `code_interpreter` 对数据进行计算分析
- 自主判断数据是否充分，决定搜索策略

### 数据来源标注

Prompt 中要求模型在报告中标注每条信息的来源：
- `[API]` — 来自项目业务数据
- `[Web]` — 来自网络搜索
- `[Computed]` — 模型计算/推导得出

## 7. Output Engine

### 模型自主决策逻辑

通过 prompt 指导模型根据内容特征选择格式：

| 内容特征 | 推荐格式 |
|---------|---------|
| 数据密集、需要交互筛选 | HTML (交互式图表，plotly) |
| 正式分析报告 | PDF (从 HTML 转换) |
| 多维数据对比 | HTML + 嵌入式图表 |
| 简短回答 | Markdown（直接返回，不生成文件） |
| 混合内容 | HTML 主报告 + PDF 摘要 |

### 渲染执行

Deep Research 的 `code_interpreter` 已经能生成部分可视化输出。框架额外处理：

1. **HTML 报告**：模型生成完整 HTML，框架直接保存
2. **PDF 转换**：在隔离 Docker 容器中用 WeasyPrint 将 HTML 转 PDF
3. **降级策略**：渲染失败时输出 Markdown + 错误日志

### 沙箱执行（仅用于额外渲染）

使用 Docker 隔离容器执行渲染代码：
- 网络：`--network=none`（完全禁止）
- 文件系统：只挂载 `/output/{task_id}/` 目录（read-write）
- 资源限制：内存 512MB，CPU 1 核
- 执行超时：60 秒
- 预装库：matplotlib, plotly, pandas, jinja2, weasyprint, pillow

## 8. Tech Stack

| 组件 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI | 异步支持，自动生成 API 文档 |
| 研究引擎 | OpenAI Python SDK | Deep Research API (o3/o4-mini) |
| 数据存储 | OpenAI Vector Store | file_search 检索业务数据 |
| 任务管理 | asyncio + SQLite | 轻量持久化，服务重启不丢失 |
| PDF 渲染 | WeasyPrint | HTML → PDF 转换（隔离容器内） |
| HTTP 客户端 | httpx | 调用项目 API |
| 部署 | Docker + docker-compose | 容器化部署 |
| 日志 | structlog | 结构化日志 |

## 9. Project Structure

```
deep-research-framework/
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.renderer           # 渲染沙箱容器镜像
├── pyproject.toml
├── .env.example
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app 入口
│   ├── config.py                 # 全局配置加载
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py             # API 路由
│   │   └── models.py             # Pydantic 请求/响应模型
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── research.py           # 研究引擎（调用 OpenAI API）
│   │   ├── data_preparation.py   # 数据预拉取 & Vector Store 管理
│   │   └── prompt_builder.py     # Prompt 构建
│   ├── output/
│   │   ├── __init__.py
│   │   └── renderer.py           # 输出渲染（Docker 隔离执行）
│   └── task/
│       ├── __init__.py
│       └── manager.py            # 任务状态管理（SQLite）
├── projects/                     # 项目配置目录
│   └── example/
│       ├── config.yaml
│       ├── api_docs/
│       └── output_prefs.yaml
├── output/                       # 生成的结果文件
├── tests/
│   ├── test_api.py
│   ├── test_research.py
│   ├── test_renderer.py
│   └── test_data_preparation.py
└── docs/
```

## 10. Environment Variables

```bash
# === 必填 ===
OPENAI_API_KEY=sk-xxx                      # OpenAI API 密钥

# === 服务配置 ===
HOST=0.0.0.0                               # 监听地址
PORT=8000                                   # 监听端口
API_TOKEN=your-service-auth-token           # 服务访问认证 Token

# === 可选 ===
STORAGE_PATH=/data/output                   # 结果存储路径，默认 ./output
PROJECTS_PATH=/data/projects                # 项目配置路径，默认 ./projects
LOG_LEVEL=info                              # 日志级别
DEFAULT_MODEL=o3-deep-research              # 默认研究模型
MAX_CONCURRENT_TASKS=5                      # 最大并发研究任务数
OUTPUT_RETENTION_DAYS=30                    # 输出文件保留天数

# === 项目 API 密钥（按需添加）===
ADMAPIX_API_KEY=xxx                         # admapix 项目 API 密钥
# OTHER_PROJECT_API_KEY=xxx
```

## 11. Security

| 关注点 | 措施 |
|-------|------|
| 服务认证 | 所有端点（除 /health）需 Bearer Token |
| task_id | UUID v4，不可预测 |
| 项目 API 密钥 | 环境变量引用，不硬编码 |
| 沙箱执行 | Docker 隔离容器，--network=none，资源限制 |
| callback_url | 仅 HTTPS，过滤私有 IP 段（10.x, 172.16-31.x, 192.168.x） |
| 并发控制 | 信号量限制同时运行的任务数 |

## 12. Error Handling

| 场景 | 处理方式 |
|------|---------|
| 项目不存在 | 返回 404，提示可用项目列表 |
| 并发超限 | 返回 429，提示重试 |
| 项目 API 预拉取失败 | 记录警告，继续研究（仅用 web search） |
| OpenAI API 失败 | 重试 2 次（指数退避），仍失败则标记任务 failed |
| 渲染代码执行失败 | 降级为 Markdown 输出，附带错误信息 |
| 执行超时 | 返回已完成的部分结果 + 超时提示 |
| 任务取消 | 标记 cancelled，尽力中断 OpenAI 轮询 |

## 13. Future Considerations

暂不实现，但设计时预留扩展空间：

- **多模型支持**：切换到 Claude Agent SDK 等
- **自定义 MCP Server**：将项目 API 封装为 MCP server，替代 prefetch + file_search 方案，实现真正的实时数据查询
- **流式进度推送**：WebSocket / SSE 实时推送研究进度
- **结果缓存**：相似问题复用历史研究结果
- **多租户**：API Token 区分不同调用方的权限和配额
- **按需数据拉取**：研究过程中模型通过 MCP 实时调用项目 API（待 OpenAI 开放更灵活的自定义工具支持）
