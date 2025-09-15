# Complete Implementation Guide: Qwen Coder CLI → Autonomous Trading Research Platform

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Development Environment Setup](#3-development-environment-setup)
4. [Phase 1: Core Infrastructure](#4-phase-1-core-infrastructure)
5. [Phase 2: Data Pipeline](#5-phase-2-data-pipeline)
6. [Phase 3: Research & Discovery](#6-phase-3-research--discovery)
7. [Phase 4: ML/AI Integration](#7-phase-4-mlai-integration)
8. [Phase 5: Backtesting Suite](#8-phase-5-backtesting-suite)
9. [Phase 6: Risk & Portfolio Management](#9-phase-6-risk--portfolio-management)
10. [Phase 7: Monitoring & Observability](#10-phase-7-monitoring--observability)
11. [Phase 8: Self-Upgrading System](#11-phase-8-self-upgrading-system)
12. [Testing & Validation](#12-testing--validation)
13. [Deployment Guide](#13-deployment-guide)

---

## 1. Project Overview

### 1.1 Goal
Transform Qwen Coder CLI into a fully autonomous cryptocurrency futures trading research platform that can:
- Discover strategies through web research and AI analysis
- Backtest and optimize strategies using multiple frameworks
- Self-improve through continuous learning
- Manage risk and portfolio optimization
- **NOT perform live trading** (research and backtesting only)

### 1.2 Key Principles
- **Modular Architecture**: Each component can be developed/tested independently
- **API-First Design**: All components communicate via well-defined APIs
- **Containerized Deployment**: Docker for consistency across environments
- **Event-Driven**: Async processing for scalability
- **Self-Documenting**: Auto-generated documentation for all strategies

---

## 2. System Architecture

### 2.1 High-Level Architecture Diagram

```mermaid
graph TB
    subgraph "User Interface Layer"
        CLI[Terminal CLI]
        WEB[Web Dashboard]
        API[REST/WebSocket API]
    end
    
    subgraph "Core Agent Layer"
        CONV[Conversation Engine]
        ORCH[Orchestrator]
        TASK[Task Manager]
    end
    
    subgraph "Research Layer"
        CRAWL[Web Crawlers]
        LLM[LLM Agents]
        KNOW[Knowledge Graph]
    end
    
    subgraph "Data Layer"
        EXCH[Exchange APIs]
        TS[Time Series DB]
        VEC[Vector DB]
    end
    
    subgraph "Analytics Layer"
        ML[ML Pipeline]
        BT[Backtesting]
        OPT[Optimization]
    end
    
    subgraph "Infrastructure"
        MON[Monitoring]
        LOG[Logging]
        QUEUE[Message Queue]
    end
```

### 2.2 Technology Stack

| Component | Primary Choice | Alternative | Purpose |
|-----------|---------------|-------------|---------|
| **Language** | Python 3.11+ | - | Main development language |
| **Async Framework** | asyncio + FastAPI | aiohttp | Async operations & API |
| **Task Queue** | Celery + Redis | RQ | Background task processing |
| **Message Bus** | Apache Kafka | RabbitMQ | Event streaming |
| **Container** | Docker + K8s | Docker Swarm | Orchestration |
| **CI/CD** | GitHub Actions | GitLab CI | Automation |

---

## 3. Development Environment Setup

### 3.1 Prerequisites Installation Script

```bash
#!/bin/bash
# setup_environment.sh

# System dependencies
sudo apt-get update
sudo apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    docker.io \
    docker-compose \
    postgresql-client \
    redis-tools \
    build-essential \
    libpq-dev \
    libta-lib0-dev \
    wget \
    curl

# Install Miniconda for environment management
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
export PATH="$HOME/miniconda3/bin:$PATH"

# Create project structure
mkdir -p qwen-trading-platform/{
    src/{core,data,research,ml,backtesting,risk,monitoring,utils},
    config,
    tests,
    docs,
    scripts,
    docker,
    notebooks
}

# Create Python virtual environment
conda create -n qwen-trading python=3.11 -y
conda activate qwen-trading
```

### 3.2 Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Databases
  postgres:
    image: timescale/timescaledb:latest-pg14
    environment:
      POSTGRES_DB: trading
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  neo4j:
    image: neo4j:5-enterprise
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
      NEO4J_ACCEPT_LICENSE_AGREEMENT: yes
    volumes:
      - neo4j_data:/data
    ports:
      - "7474:7474"
      - "7687:7687"

  influxdb:
    image: influxdb:2.7
    environment:
      DOCKER_INFLUXDB_INIT_MODE: setup
      DOCKER_INFLUXDB_INIT_USERNAME: admin
      DOCKER_INFLUXDB_INIT_PASSWORD: ${INFLUX_PASSWORD}
      DOCKER_INFLUXDB_INIT_ORG: trading
      DOCKER_INFLUXDB_INIT_BUCKET: metrics
    volumes:
      - influxdb_data:/var/lib/influxdb2
    ports:
      - "8086:8086"

  questdb:
    image: questdb/questdb:latest
    volumes:
      - questdb_data:/var/lib/questdb
    ports:
      - "9000:9000"
      - "8812:8812"

  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"

  # Message Queue
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
    ports:
      - "9092:9092"

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
    ports:
      - "2181:2181"

  # ML Tools
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    command: mlflow server --host 0.0.0.0 --backend-store-uri postgresql://trader:${DB_PASSWORD}@postgres/mlflow --default-artifact-root s3://mlflow
    ports:
      - "5000:5000"

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"

  # Development Tools
  jupyter:
    image: jupyter/datascience-notebook:latest
    environment:
      JUPYTER_TOKEN: ${JUPYTER_TOKEN}
    volumes:
      - ./notebooks:/home/jovyan/work
    ports:
      - "8888:8888"

volumes:
  postgres_data:
  neo4j_data:
  influxdb_data:
  questdb_data:
  qdrant_data:
  prometheus_data:
  grafana_data:
```

### 3.3 Python Dependencies

```toml
# pyproject.toml
[project]
name = "qwen-trading-platform"
version = "2.0.0"
requires-python = ">=3.11"

[project.dependencies]
# Core
fastapi = "^0.104.0"
uvicorn = "^0.24.0"
pydantic = "^2.4.0"
typer = "^0.9.0"
rich = "^13.6.0"

# Async & Concurrency
asyncio = "*"
aiohttp = "^3.9.0"
celery = "^5.3.0"
redis = "^5.0.0"
aiokafka = "^0.10.0"

# Data Management
pandas = "^2.1.0"
numpy = "^1.26.0"
polars = "^0.19.0"
duckdb = "^0.9.0"
sqlalchemy = "^2.0.0"
asyncpg = "^0.29.0"

# Databases
neo4j = "^5.14.0"
influxdb-client = "^1.38.0"
qdrant-client = "^1.7.0"
psycopg2-binary = "^2.9.0"

# Web Scraping & Research
crawl4ai = "^0.2.0"
firecrawl-py = "^0.0.16"
scrapegraphai = "^1.2.0"
beautifulsoup4 = "^4.12.0"
selenium = "^4.15.0"
playwright = "^1.40.0"

# Exchange APIs
ccxt = "^4.1.0"
python-binance = "^1.0.0"

# Technical Analysis
ta-lib = "^0.4.28"
pandas-ta = "^0.3.14"
technical = "^1.4.0"
tsfresh = "^0.20.0"
feature-engine = "^1.6.0"

# Machine Learning
scikit-learn = "^1.3.0"
optuna = "^3.4.0"
mlflow = "^2.8.0"
flaml = "^2.1.0"
autogluon = "^1.0.0"
shap = "^0.44.0"

# Deep Learning
torch = "^2.1.0"
tensorflow = "^2.15.0"
transformers = "^4.35.0"
gluonts = "^0.14.0"
darts = "^0.27.0"
neuralprophet = "^0.8.0"

# Reinforcement Learning
stable-baselines3 = "^2.2.0"
tensorforce = "^0.6.5"

# Backtesting
freqtrade = "^2023.10"
vectorbt = "^0.26.0"
backtrader = "^1.9.78"
pyalgotrade = "^0.20"
zipline-reloaded = "^3.0.0"

# Portfolio & Risk
pyportfolioopt = "^1.5.0"
riskfolio-lib = "^5.0.0"
quantstats = "^0.0.62"
empyrical = "^0.5.5"

# LLM & NLP
langchain = "^0.1.0"
llama-index = "^0.9.0"
openai = "^1.3.0"
anthropic = "^0.7.0"

# Monitoring & Logging
prometheus-client = "^0.19.0"
grafana-api = "^1.0.3"
loguru = "^0.7.0"
sentry-sdk = "^1.38.0"

# Visualization
streamlit = "^1.29.0"
gradio = "^4.8.0"
plotly = "^5.18.0"
bokeh = "^3.3.0"

# Testing
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"
pytest-cov = "^4.1.0"
hypothesis = "^6.92.0"

# Development
black = "^23.11.0"
ruff = "^0.1.6"
mypy = "^1.7.0"
pre-commit = "^3.5.0"
```

---

## 4. Phase 1: Core Infrastructure (Week 1-2)

### 4.1 Project Structure

```python
# src/core/__init__.py
"""Core infrastructure components"""

from .config import Config
from .orchestrator import Orchestrator
from .conversation import ConversationEngine
from .task_manager import TaskManager

__all__ = ['Config', 'Orchestrator', 'ConversationEngine', 'TaskManager']
```

### 4.2 Configuration Management

```python
# src/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Config(BaseSettings):
    """Central configuration management"""
    
    # Application
    app_name: str = "Qwen Trading Platform"
    version: str = "2.0.0"
    debug: bool = False
    
    # API Keys (from environment)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None
    tavily_api_key: Optional[str] = None
    
    # Database URLs
    postgres_url: str = "postgresql://trader:password@localhost/trading"
    neo4j_url: str = "bolt://localhost:7687"
    influxdb_url: str = "http://localhost:8086"
    questdb_url: str = "postgresql://localhost:8812/trading"
    qdrant_url: str = "http://localhost:6333"
    
    # Redis & Kafka
    redis_url: str = "redis://localhost:6379"
    kafka_bootstrap_servers: str = "localhost:9092"
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    
    # Paths
    data_dir: str = "./data"
    models_dir: str = "./models"
    logs_dir: str = "./logs"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Singleton instance
config = Config()
```

### 4.3 Orchestrator Implementation

```python
# src/core/orchestrator.py
import asyncio
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import uuid
from datetime import datetime
from loguru import logger

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    id: str
    type: str
    payload: Dict[str, Any]
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    result: Optional[Any] = None
    error: Optional[str] = None

class Orchestrator:
    """Central orchestrator for all platform operations"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.handlers: Dict[str, Any] = {}
        self.running = False
        
    async def start(self):
        """Start the orchestrator"""
        self.running = True
        logger.info("Orchestrator started")
        
        # Start background task processor
        asyncio.create_task(self._process_tasks())
        
    async def stop(self):
        """Stop the orchestrator"""
        self.running = False
        logger.info("Orchestrator stopped")
        
    def register_handler(self, task_type: str, handler):
        """Register a task handler"""
        self.handlers[task_type] = handler
        logger.info(f"Registered handler for {task_type}")
        
    async def submit_task(self, task_type: str, payload: Dict[str, Any]) -> str:
        """Submit a new task"""
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            type=task_type,
            payload=payload,
            status=TaskStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        self.tasks[task_id] = task
        logger.info(f"Task {task_id} submitted: {task_type}")
        return task_id
        
    async def _process_tasks(self):
        """Background task processor"""
        while self.running:
            pending_tasks = [
                task for task in self.tasks.values() 
                if task.status == TaskStatus.PENDING
            ]
            
            for task in pending_tasks:
                await self._execute_task(task)
                
            await asyncio.sleep(1)
            
    async def _execute_task(self, task: Task):
        """Execute a single task"""
        try:
            task.status = TaskStatus.RUNNING
            task.updated_at = datetime.utcnow()
            
            handler = self.handlers.get(task.type)
            if not handler:
                raise ValueError(f"No handler for task type: {task.type}")
                
            result = await handler(task.payload)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.updated_at = datetime.utcnow()
            
            logger.info(f"Task {task.id} completed successfully")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.updated_at = datetime.utcnow()
            logger.error(f"Task {task.id} failed: {e}")
```

### 4.4 Conversation Engine

```python
# src/core/conversation.py
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from typing import List, Optional
import json

class ConversationEngine:
    """Enhanced conversation management with context awareness"""
    
    def __init__(self, llm_provider="openai"):
        self.llm_provider = llm_provider
        self.memory = ConversationSummaryBufferMemory(
            llm=self._get_llm(),
            max_token_limit=2000,
            return_messages=True
        )
        self.context = {}
        
    def _get_llm(self):
        """Get LLM instance based on provider"""
        if self.llm_provider == "openai":
            from langchain.llms import OpenAI
            return OpenAI(temperature=0.7)
        elif self.llm_provider == "anthropic":
            from langchain.llms import Anthropic
            return Anthropic()
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
            
    async def process_message(self, message: str) -> str:
        """Process user message and generate response"""
        # Add to memory
        self.memory.chat_memory.add_user_message(message)
        
        # Determine intent and route accordingly
        intent = await self._analyze_intent(message)
        
        # Generate response based on intent
        response = await self._generate_response(message, intent)
        
        # Add response to memory
        self.memory.chat_memory.add_ai_message(response)
        
        return response
        
    async def _analyze_intent(self, message: str) -> Dict[str, Any]:
        """Analyze user intent from message"""
        # This would use NLP/LLM to determine intent
        # For now, simple keyword matching
        intents = {
            "research": ["research", "find", "discover", "search"],
            "backtest": ["backtest", "test", "evaluate", "performance"],
            "optimize": ["optimize", "improve", "enhance", "tune"],
            "analyze": ["analyze", "analysis", "examine", "investigate"]
        }
        
        message_lower = message.lower()
        for intent, keywords in intents.items():
            if any(keyword in message_lower for keyword in keywords):
                return {"type": intent, "confidence": 0.8}
                
        return {"type": "general", "confidence": 0.5}
        
    async def _generate_response(self, message: str, intent: Dict[str, Any]) -> str:
        """Generate response based on message and intent"""
        # Route to appropriate handler based on intent
        if intent["type"] == "research":
            return await self._handle_research_request(message)
        elif intent["type"] == "backtest":
            return await self._handle_backtest_request(message)
        elif intent["type"] == "optimize":
            return await self._handle_optimize_request(message)
        else:
            return await self._handle_general_query(message)
```

---

## 5. Phase 2: Data Pipeline (Week 3-4)

### 5.1 Exchange Data Manager

```python
# src/data/exchange_manager.py
import ccxt
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
from loguru import logger

class ExchangeDataManager:
    """Manage data from multiple exchanges"""
    
    def __init__(self):
        self.exchanges = {}
        self.initialize_exchanges()
        
    def initialize_exchanges(self):
        """Initialize exchange connections"""
        # Binance
        self.exchanges['binance'] = ccxt.binance({
            'apiKey': config.binance_api_key,
            'secret': config.binance_api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        # Add more exchanges as needed
        self.exchanges['bybit'] = ccxt.bybit({'enableRateLimit': True})
        self.exchanges['okx'] = ccxt.okx({'enableRateLimit': True})
        
    async def fetch_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Fetch OHLCV data from exchange"""
        try:
            exchange_obj = self.exchanges.get(exchange)
            if not exchange_obj:
                raise ValueError(f"Exchange {exchange} not supported")
                
            # Convert datetime to timestamp
            since_ts = int(since.timestamp() * 1000) if since else None
            
            # Fetch data
            ohlcv = await exchange_obj.fetch_ohlcv(
                symbol,
                timeframe,
                since=since_ts,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} candles for {symbol} from {exchange}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from {exchange}: {e}")
            raise
            
    async def fetch_order_book(
        self,
        exchange: str,
        symbol: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Fetch order book data"""
        exchange_obj = self.exchanges.get(exchange)
        if not exchange_obj:
            raise ValueError(f"Exchange {exchange} not supported")
            
        return await exchange_obj.fetch_order_book(symbol, limit)
        
    async def fetch_funding_rate(
        self,
        exchange: str,
        symbol: str
    ) -> float:
        """Fetch current funding rate for futures"""
        exchange_obj = self.exchanges.get(exchange)
        if not exchange_obj:
            raise ValueError(f"Exchange {exchange} not supported")
            
        funding = await exchange_obj.fetch_funding_rate(symbol)
        return funding['fundingRate']
```

### 5.2 Database Managers

```python
# src/data/database_manager.py
from neo4j import AsyncGraphDatabase
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import ASYNCHRONOUS
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import asyncpg
import duckdb
from typing import Dict, Any, List
import numpy as np

class DatabaseManager:
    """Unified database management"""
    
    def __init__(self):
        self.neo4j = None
        self.influxdb = None
        self.qdrant = None
        self.postgres = None
        self.duckdb = None
        
    async def initialize(self):
        """Initialize all database connections"""
        await self._init_neo4j()
        await self._init_influxdb()
        await self._init_qdrant()
        await self._init_postgres()
        await self._init_duckdb()
        
    async def _init_neo4j(self):
        """Initialize Neo4j connection"""
        self.neo4j = AsyncGraphDatabase.driver(
            config.neo4j_url,
            auth=("neo4j", config.neo4j_password)
        )
        
        # Create constraints and indexes
        async with self.neo4j.session() as session:
            await session.run("""
                CREATE CONSTRAINT strategy_id IF NOT EXISTS
                FOR (s:Strategy) REQUIRE s.id IS UNIQUE
            """)
            await session.run("""
                CREATE INDEX strategy_performance IF NOT EXISTS
                FOR (s:Strategy) ON (s.sharpe_ratio)
            """)
            
    async def _init_influxdb(self):
        """Initialize InfluxDB connection"""
        self.influxdb = InfluxDBClient(
            url=config.influxdb_url,
            token=config.influxdb_token,
            org=config.influxdb_org
        )
        self.influx_write_api = self.influxdb.write_api(write_options=ASYNCHRONOUS)
        
    async def _init_qdrant(self):
        """Initialize Qdrant vector database"""
        self.qdrant = QdrantClient(url=config.qdrant_url)
        
        # Create collections if they don't exist
        collections = ["strategies", "research_papers", "patterns"]
        for collection in collections:
            try:
                self.qdrant.get_collection(collection)
            except:
                self.qdrant.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
                
    async def _init_postgres(self):
        """Initialize PostgreSQL/TimescaleDB connection"""
        self.postgres = await asyncpg.create_pool(
            config.postgres_url,
            min_size=10,
            max_size=20
        )
        
        # Create tables
        async with self.postgres.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS price_data (
                    exchange VARCHAR(50),
                    symbol VARCHAR(50),
                    timestamp TIMESTAMPTZ NOT NULL,
                    open DECIMAL,
                    high DECIMAL,
                    low DECIMAL,
                    close DECIMAL,
                    volume DECIMAL,
                    PRIMARY KEY (exchange, symbol, timestamp)
                );
                
                -- Convert to hypertable for TimescaleDB
                SELECT create_hypertable('price_data', 'timestamp', 
                    if_not_exists => TRUE);
            """)
            
    async def _init_duckdb(self):
        """Initialize DuckDB for analytics"""
        self.duckdb = duckdb.connect(':memory:')
        
        # Create analytical views
        self.duckdb.execute("""
            CREATE VIEW IF NOT EXISTS strategy_performance AS
            SELECT 
                strategy_id,
                AVG(returns) as avg_returns,
                STDDEV(returns) as volatility,
                AVG(returns) / NULLIF(STDDEV(returns), 0) as sharpe_ratio
            FROM backtest_results
            GROUP BY strategy_id
        """)
```

### 5.3 Alternative Data Sources

```python
# src/data/alternative_data.py
import aiohttp
from typing import Dict, Any
import pandas as pd

class AlternativeDataManager:
    """Manage alternative data sources"""
    
    def __init__(self):
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
        
    async def fetch_fear_greed_index(self) -> Dict[str, Any]:
        """Fetch crypto fear and greed index"""
        url = "https://api.alternative.me/fng/"
        async with self.session.get(url) as response:
            data = await response.json()
            return {
                'value': int(data['data'][0]['value']),
                'classification': data['data'][0]['value_classification'],
                'timestamp': data['data'][0]['timestamp']
            }
            
    async def fetch_defi_tvl(self, protocol: str = None) -> Dict[str, Any]:
        """Fetch DeFi TVL data from DeFiLlama"""
        if protocol:
            url = f"https://api.llama.fi/protocol/{protocol}"
        else:
            url = "https://api.llama.fi/protocols"
            
        async with self.session.get(url) as response:
            return await response.json()
            
    async def fetch_coinglass_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch futures data from Coinglass"""
        # Note: Coinglass requires API key
        headers = {'X-API-KEY': config.coinglass_api_key}
        url = f"https://api.coinglass.com/api/futures/openInterest/chart"
        params = {'symbol': symbol, 'interval': '1h'}
        
        async with self.session.get(url, headers=headers, params=params) as response:
            return await response.json()
```

---

## 6. Phase 3: Research & Discovery (Week 5-6)

### 6.1 Web Scraping Integration

```python
# src/research/web_scraper.py
from crawl4ai import AsyncWebCrawler
from firecrawl import FirecrawlApp
from scrap