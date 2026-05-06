# HAPPO-GNN-RL 电动汽车充电站动态调度优化 Web 系统

## 🚀 快速启动

### 方式一：Docker（推荐）

```bash
# 构建并启动（需要将 data.csv 放到项目根目录）
docker-compose up --build -d

# 访问系统
open http://localhost:5000
```

### 方式二：本地运行

```bash
cd webapp
pip install -r requirements.txt
python app.py

# 访问系统
open http://localhost:5000
```

## 👤 登录账户

| 角色 | 用户名 | 密码 |
|------|--------|------|
| 管理员 | admin | admin123 |
| 普通用户 | user | user123 |

## 📋 功能模块

1. **控制面板** — 系统整体数据统计 + 等待时间/能耗双轴折线图
2. **充电站管理** — 查看所有充电站基础信息
3. **充电记录** — 查看历史充电会话数据（最多100条）
4. **调度决策** — 调用 HAPPO-GNN-RL 模型，输入车辆/队列/电价/可再生能源比例，输出最优调度决策
5. **方法对比** — 对比 6 种调度策略的性能指标

## 🏗️ 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Flask 3.x + Flask-Login + Flask-SQLAlchemy |
| 数据库 | SQLite（charging.db） |
| 前端 | HTML5 + CSS3 + Vanilla JS + Chart.js |
| 模型推理 | HAPPO-GNN-RL（规则代理模式） |
| 容器化 | Docker + Docker Compose |

## 📂 目录结构

```
webapp/
├── app.py              # Flask 主程序（含所有路由和 API）
├── requirements.txt    # Python 依赖
├── Dockerfile          # Docker 构建文件
├── .dockerignore       # Docker 忽略文件
├── .env.example        # 环境变量示例
├── templates/          # HTML 模板
│   ├── index.html      # 主仪表盘
│   ├── login.html      # 登录页
│   └── register.html   # 注册页
└── static/             # 静态资源
    ├── style.css       # 样式表
    └── app.js         # 前端交互逻辑
```

## 🔧 数据初始化

系统启动时自动执行 `init_db()`：
- 创建默认管理员/用户账户
- 从项目根目录 `data.csv` 读取数据
- 导入充电站信息到 `stations` 表
- 导入充电会话到 `charging_sessions` 表（前1000条）

## 📡 API 端点

| 方法 | 路由 | 说明 |
|------|------|------|
| GET | `/api/stats` | 系统统计摘要 |
| GET | `/api/stations` | 充电站列表 |
| GET | `/api/sessions?limit=N` | 充电记录（最新N条）|
| POST | `/api/predict` | HAPPO-GNN-RL 调度决策 |
| GET | `/api/compare` | 多方法对比数据 |

## ⚠️ 模型说明

当前 `/api/predict` 使用**规则代理模式**：
- 基于车辆类型（Bus/Car/Two-Wheeler）设置优先级
- 基于队列长度和电价动态调整功率
- 实际项目应加载已训练的 PyTorch 模型权重（`HAPPO_GNN_RL_Model.py`）
