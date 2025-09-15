---

## 9. Phase 6: Risk & Portfolio Management (Week 11)

### 9.1 Advanced Risk Management

```python
# src/risk/risk_manager.py
import numpy as np
import pandas as pd
from scipy import stats
import scipy.optimize as opt
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class RiskMetric(Enum):
    VAR = "value_at_risk"
    CVAR = "conditional_var"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE = "sharpe_ratio"
    SORTINO = "sortino_ratio"
    CALMAR = "calmar_ratio"
    BETA = "beta"
    ALPHA = "alpha"

@dataclass
class RiskLimits:
    max_position_size: float = 0.1  # 10% max per position
    max_portfolio_var: float = 0.05  # 5% daily VaR
    max_drawdown: float = 0.15  # 15% max drawdown
    min_sharpe_ratio: float = 1.0
    max_correlation: float = 0.7
    max_leverage: float = 2.0

class RiskManager:
    """Advanced risk management and portfolio optimization"""
    
    def __init__(self, risk_limits: RiskLimits = None):
        self.risk_limits = risk_limits or RiskLimits()
        self.portfolio_metrics = {}
        
    def calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = "historical"
    ) -> float:
        """Calculate Value at Risk"""
        
        if method == "historical":
            return np.percentile(returns.dropna(), (1 - confidence_level) * 100)
            
        elif method == "parametric":
            mu = returns.mean()
            sigma = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            return mu + z_score * sigma
            
        elif method == "monte_carlo":
            # Fit distribution and simulate
            mu = returns.mean()
            sigma = returns.std()
            simulated = np.random.normal(mu, sigma, 10000)
            return np.percentile(simulated, (1 - confidence_level) * 100)
            
        else:
            raise ValueError(f"Unknown VaR method: {method}")
            
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.calculate_var(returns, confidence_level, "historical")
        return returns[returns <= var].mean()
        
    def calculate_maximum_drawdown(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate maximum drawdown and related metrics"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        max_dd = drawdown.min()
        max_dd_start = drawdown.idxmin()
        
        # Find recovery date
        recovery_date = None
        for date in drawdown[max_dd_start:].index:
            if drawdown[date] >= -0.001:  # Within 0.1% of recovery
                recovery_date = date
                break
                
        recovery_days = None
        if recovery_date:
            recovery_days = (recovery_date - max_dd_start).days
            
        return {
            'max_drawdown': max_dd,
            'drawdown_start': max_dd_start,
            'recovery_date': recovery_date,
            'recovery_days': recovery_days,
            'current_drawdown': drawdown.iloc[-1]
        }
        
    def calculate_risk_adjusted_returns(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        benchmark_returns: pd.Series = None
    ) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics"""
        
        metrics = {}
        
        # Basic statistics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = (1 + returns.mean()) ** 252 - 1
        metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        excess_returns = returns - risk_free_rate / 252
        metrics['sharpe_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns### 7.3 Deep Learning Integration

```python
# src/ml/deep_learning.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
import mlflow

class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series data"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 60,
        target_col: str = 'close',
        feature_cols: List[str] = None
    ):
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.feature_cols = feature_cols or [col for col in data.columns if col != target_col]
        
        # Scale data
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        self.features = self.feature_scaler.fit_transform(data[self.feature_cols])
        self.targets = self.target_scaler.fit_transform(data[[target_col]])
        
    def __len__(self):
        return len(self.features) - self.sequence_length
        
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class LSTMPredictor(pl.LightningModule):
    """LSTM-based price prediction model"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        output = self.dropout(last_output)
        return self.linear(output)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

class TransformerPredictor(pl.LightningModule):
    """Transformer-based price prediction model"""
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        learning_rate: float = 0.001
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer expects (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Apply transformer
        output = self.transformer_encoder(x)
        
        # Take the last output and convert back to (batch_size, d_model)
        output = output[-1]
        
        # Apply dropout and final linear layer
        output = self.dropout(output)
        return self.linear(output)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class DeepLearningManager:
    """Manage deep learning models for trading"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    async def train_lstm_model(
        self,
        data: pd.DataFrame,
        sequence_length: int = 60,
        hidden_size: int = 128,
        num_layers: int = 2,
        epochs: int = 100
    ) -> Dict[str, Any]:
        """Train LSTM model for price prediction"""
        
        with mlflow.start_run(run_name="lstm_training"):
            # Prepare dataset
            dataset = TimeSeriesDataset(data, sequence_length=sequence_length)
            
            # Split data
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Initialize model
            input_size = len(dataset.feature_cols)
            model = LSTMPredictor(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers
            )
            
            # Train model
            trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10)
            trainer.fit(model, train_loader, val_loader)
            
            # Log parameters
            mlflow.log_param("sequence_length", sequence_length)
            mlflow.log_param("hidden_size", hidden_size)
            mlflow.log_param("num_layers", num_layers)
            mlflow.log_param("input_size", input_size)
            
            # Store model and scaler
            self.models['lstm'] = model
            self.scalers['lstm'] = {
                'feature_scaler': dataset.feature_scaler,
                'target_scaler': dataset.target_scaler
            }
            
            return {
                'model_type': 'lstm',
                'input_size': input_size,
                'sequence_length': sequence_length,
                'train_size': train_size,
                'val_size': val_size
            }
            
    async def train_transformer_model(
        self,
        data: pd.DataFrame,
        sequence_length: int = 60,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        epochs: int = 100
    ) -> Dict[str, Any]:
        """Train Transformer model for price prediction"""
        
        with mlflow.start_run(run_name="transformer_training"):
            # Prepare dataset
            dataset = TimeSeriesDataset(data, sequence_length=sequence_length)
            
            # Split data
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Initialize model
            input_size = len(dataset.feature_cols)
            model = TransformerPredictor(
                input_size=input_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers
            )
            
            # Train model
            trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10)
            trainer.fit(model, train_loader, val_loader)
            
            # Log parameters
            mlflow.log_param("sequence_length", sequence_length)
            mlflow.log_param("d_model", d_model)
            mlflow.log_param("nhead", nhead)
            mlflow.log_param("num_layers", num_layers)
            mlflow.log_param("input_size", input_size)
            
            # Store model and scaler
            self.models['transformer'] = model
            self.scalers['transformer'] = {
                'feature_scaler': dataset.feature_scaler,
                'target_scaler': dataset.target_scaler
            }
            
            return {
                'model_type': 'transformer',
                'input_size': input_size,
                'sequence_length': sequence_length,
                'train_size': train_size,
                'val_size': val_size
            }
```

---

## 8. Phase 5: Backtesting Suite (Week 9-10)

### 8.1 FreqTrade Integration

```python
# src/backtesting/freqtrade_manager.py
import subprocess
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import shutil
from jinja2 import Template
import asyncio

class FreqTradeManager:
    """Manage FreqTrade backtesting and optimization"""
    
    def __init__(self):
        self.freqtrade_path = "freqtrade"
        self.strategies_dir = Path("user_data/strategies")
        self.config_dir = Path("user_data/config")
        self.results_dir = Path("user_data/backtest_results")
        
        # Create directories
        self.strategies_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_strategy_file(
        self,
        strategy_name: str,
        entry_signals: List[Dict],
        exit_signals: List[Dict],
        indicators: List[Dict],
        stoploss: float = -0.05,
        roi: Dict = None
    ) -> Path:
        """Generate FreqTrade strategy file"""
        
        roi = roi or {
            "0": 0.10,
            "10": 0.05,
            "20": 0.02,
            "30": 0.01
        }
        
        strategy_template = Template("""
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd
from pandas import DataFrame

class {{ strategy_name }}(IStrategy):
    
    # Strategy interface version
    INTERFACE_VERSION = 3
    
    # ROI table
    minimal_roi = {{ roi }}
    
    # Stoploss
    stoploss = {{ stoploss }}
    
    # Timeframe
    timeframe = '5m'
    
    # Run "populate_indicators" only for new candle
    process_only_new_candles = False
    
    # Experimental settings
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30
    
    def informative_pairs(self):
        return []
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        {% for indicator in indicators %}
        # {{ indicator.description }}
        {% if indicator.type == 'sma' %}
        dataframe['{{ indicator.name }}'] = ta.SMA(dataframe, timeperiod={{ indicator.period }})
        {% elif indicator.type == 'ema' %}
        dataframe['{{ indicator.name }}'] = ta.EMA(dataframe, timeperiod={{ indicator.period }})
        {% elif indicator.type == 'rsi' %}
        dataframe['{{ indicator.name }}'] = ta.RSI(dataframe, timeperiod={{ indicator.period }})
        {% elif indicator.type == 'macd' %}
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']# Implementation Guide Continuation

## 6. Phase 3: Research & Discovery (Week 5-6) - Continued

### 6.1 Web Scraping Integration - Continued

```python
# src/research/web_scraper.py
from crawl4ai import AsyncWebCrawler
from firecrawl import FirecrawlApp
from scrapegraphai import SmartScraper
import aiohttp
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import asyncio
from typing import List, Dict, Any
import json

class WebScrapingManager:
    """Comprehensive web scraping for strategy discovery"""
    
    def __init__(self):
        self.crawl4ai = AsyncWebCrawler(verbose=True)
        self.firecrawl = FirecrawlApp(api_key=config.firecrawl_api_key)
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        await self.crawl4ai.astart()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        await self.crawl4ai.aclose()
        
    async def scrape_tradingview_ideas(self, asset: str = "BTCUSDT") -> List[Dict]:
        """Scrape TradingView for trading ideas"""
        url = f"https://www.tradingview.com/symbols/{asset}/ideas/"
        
        result = await self.crawl4ai.arun(
            url=url,
            word_count_threshold=10,
            extraction_strategy="LLMExtractionStrategy",
            extraction_schema={
                "type": "object",
                "properties": {
                    "ideas": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "author": {"type": "string"},
                                "strategy_type": {"type": "string"},
                                "indicators": {"type": "array", "items": {"type": "string"}},
                                "timeframe": {"type": "string"},
                                "sentiment": {"type": "string"}
                            }
                        }
                    }
                }
            }
        )
        
        return json.loads(result.extracted_content)["ideas"]
        
    async def scrape_github_strategies(self) -> List[Dict]:
        """Scrape GitHub for trading strategy repositories"""
        search_queries = [
            "cryptocurrency trading strategy python",
            "bitcoin futures backtesting",
            "crypto trading bot freqtrade",
            "quantitative trading algorithms crypto"
        ]
        
        strategies = []
        for query in search_queries:
            url = f"https://github.com/search?q={query}&type=repositories"
            
            result = await self.crawl4ai.arun(
                url=url,
                word_count_threshold=5,
                extraction_strategy="LLMExtractionStrategy",
                extraction_schema={
                    "type": "object",
                    "properties": {
                        "repositories": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "url": {"type": "string"},
                                    "stars": {"type": "integer"},
                                    "language": {"type": "string"},
                                    "last_updated": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            )
            
            if result.extracted_content:
                repos = json.loads(result.extracted_content)["repositories"]
                strategies.extend(repos)
                
        return strategies
        
    async def scrape_research_papers(self) -> List[Dict]:
        """Scrape arXiv and SSRN for trading research"""
        arxiv_queries = [
            "cryptocurrency trading",
            "algorithmic trading bitcoin",
            "machine learning financial markets",
            "high frequency trading crypto"
        ]
        
        papers = []
        for query in arxiv_queries:
            url = f"https://arxiv.org/search/?query={query}&searchtype=all"
            
            # Use Firecrawl for JavaScript-heavy sites
            try:
                result = self.firecrawl.scrape_url(
                    url,
                    params={
                        'formats': ['extract'],
                        'extract': {
                            'schema': {
                                'type': 'object',
                                'properties': {
                                    'papers': {
                                        'type': 'array',
                                        'items': {
                                            'type': 'object',
                                            'properties': {
                                                'title': {'type': 'string'},
                                                'authors': {'type': 'array'},
                                                'abstract': {'type': 'string'},
                                                'pdf_url': {'type': 'string'},
                                                'published': {'type': 'string'}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                )
                
                if result.get('extract'):
                    papers.extend(result['extract']['papers'])
                    
            except Exception as e:
                logger.error(f"Error scraping arXiv: {e}")
                
        return papers
        
    async def analyze_strategy_from_text(self, text: str) -> Dict[str, Any]:
        """Use ScrapegraphAI to analyze strategy from scraped text"""
        graph_config = {
            "llm": {
                "model": "openai/gpt-4",
                "api_key": config.openai_api_key,
            }
        }
        
        smart_scraper = SmartScraper(
            prompt="Extract trading strategy details including entry/exit rules, indicators used, risk management, and expected performance metrics",
            source=text,
            config=graph_config
        )
        
        return smart_scraper.run()
```

### 6.2 Research Agent Integration

```python
# src/research/research_agents.py
from gpt_researcher import GPTResearcher
from crewai import Agent, Task, Crew
from langchain.tools import Tool
from typing import List, Dict, Any
import asyncio

class TradingResearchAgent:
    """Autonomous research agent for trading strategies"""
    
    def __init__(self):
        self.gpt_researcher = None
        self.crew = None
        self.initialize_crew()
        
    def initialize_crew(self):
        """Initialize CrewAI agents for collaborative research"""
        
        # Strategy Researcher Agent
        strategy_researcher = Agent(
            role='Strategy Researcher',
            goal='Discover and analyze cryptocurrency trading strategies',
            backstory='Expert in quantitative finance and cryptocurrency markets',
            tools=[self._web_search_tool(), self._paper_analysis_tool()],
            verbose=True
        )
        
        # Technical Analyst Agent
        technical_analyst = Agent(
            role='Technical Analyst',
            goal='Analyze technical indicators and chart patterns',
            backstory='Experienced technical analyst specializing in crypto markets',
            tools=[self._technical_analysis_tool(), self._pattern_recognition_tool()],
            verbose=True
        )
        
        # Risk Analyst Agent
        risk_analyst = Agent(
            role='Risk Analyst',
            goal='Evaluate strategy risk and drawdown characteristics',
            backstory='Risk management expert with focus on crypto volatility',
            tools=[self._risk_calculation_tool(), self._backtesting_tool()],
            verbose=True
        )
        
        # Strategy Optimizer Agent
        optimizer = Agent(
            role='Strategy Optimizer',
            goal='Optimize strategy parameters for maximum risk-adjusted returns',
            backstory='Quantitative researcher specializing in parameter optimization',
            tools=[self._optimization_tool(), self._validation_tool()],
            verbose=True
        )
        
        self.agents = {
            'researcher': strategy_researcher,
            'analyst': technical_analyst,
            'risk': risk_analyst,
            'optimizer': optimizer
        }
        
    def _web_search_tool(self) -> Tool:
        """Web search tool for agents"""
        async def search(query: str) -> str:
            researcher = GPTResearcher(query=query, report_type="research_report")
            return await researcher.conduct_research()
            
        return Tool(
            name="web_search",
            description="Search the web for trading strategy information",
            func=search
        )
        
    def _paper_analysis_tool(self) -> Tool:
        """Academic paper analysis tool"""
        def analyze_paper(paper_url: str) -> str:
            # Implementation for paper analysis
            return f"Analysis of paper: {paper_url}"
            
        return Tool(
            name="paper_analysis",
            description="Analyze academic papers for trading insights",
            func=analyze_paper
        )
        
    def _technical_analysis_tool(self) -> Tool:
        """Technical analysis tool"""
        def technical_analysis(symbol: str, timeframe: str) -> str:
            # Implementation for technical analysis
            return f"Technical analysis for {symbol} on {timeframe}"
            
        return Tool(
            name="technical_analysis",
            description="Perform technical analysis on price data",
            func=technical_analysis
        )
        
    def _pattern_recognition_tool(self) -> Tool:
        """Pattern recognition tool"""
        def recognize_patterns(data: str) -> str:
            # Implementation for pattern recognition
            return "Recognized patterns in data"
            
        return Tool(
            name="pattern_recognition",
            description="Recognize chart patterns and formations",
            func=recognize_patterns
        )
        
    def _risk_calculation_tool(self) -> Tool:
        """Risk calculation tool"""
        def calculate_risk(strategy_data: str) -> str:
            # Implementation for risk calculations
            return "Risk metrics calculated"
            
        return Tool(
            name="risk_calculation",
            description="Calculate various risk metrics for strategies",
            func=calculate_risk
        )
        
    def _backtesting_tool(self) -> Tool:
        """Backtesting tool"""
        def backtest_strategy(strategy: str) -> str:
            # Implementation for backtesting
            return "Backtest completed"
            
        return Tool(
            name="backtesting",
            description="Backtest trading strategies",
            func=backtest_strategy
        )
        
    def _optimization_tool(self) -> Tool:
        """Optimization tool"""
        def optimize_parameters(strategy: str) -> str:
            # Implementation for optimization
            return "Parameters optimized"
            
        return Tool(
            name="optimization",
            description="Optimize strategy parameters",
            func=optimize_parameters
        )
        
    def _validation_tool(self) -> Tool:
        """Validation tool"""
        def validate_strategy(strategy: str) -> str:
            # Implementation for validation
            return "Strategy validated"
            
        return Tool(
            name="validation",
            description="Validate strategy robustness",
            func=validate_strategy
        )
        
    async def research_strategy_topic(self, topic: str) -> Dict[str, Any]:
        """Research a specific trading strategy topic"""
        
        # Define research task
        research_task = Task(
            description=f"Research and analyze {topic} trading strategies for cryptocurrency futures",
            agent=self.agents['researcher'],
            expected_output="Comprehensive report on strategy approaches, indicators, and performance characteristics"
        )
        
        # Define analysis task
        analysis_task = Task(
            description=f"Perform technical analysis on {topic} strategies",
            agent=self.agents['analyst'],
            expected_output="Technical analysis report with indicator combinations and signal quality"
        )
        
        # Define risk assessment task
        risk_task = Task(
            description=f"Assess risk characteristics of {topic} strategies",
            agent=self.agents['risk'],
            expected_output="Risk assessment report with drawdown analysis and risk metrics"
        )
        
        # Define optimization task
        optimization_task = Task(
            description=f"Optimize parameters for {topic} strategies",
            agent=self.agents['optimizer'],
            expected_output="Optimized strategy parameters with performance improvements"
        )
        
        # Create crew and execute
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=[research_task, analysis_task, risk_task, optimization_task],
            verbose=True
        )
        
        result = crew.kickoff()
        
        return {
            'topic': topic,
            'research_results': result,
            'timestamp': datetime.utcnow(),
            'agents_involved': list(self.agents.keys())
        }
```

### 6.3 Knowledge Graph Integration

```python
# src/research/knowledge_graph.py
from neo4j import AsyncGraphDatabase
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import numpy as np

class TradingKnowledgeGraph:
    """Neo4j-based knowledge graph for trading strategies"""
    
    def __init__(self):
        self.driver = AsyncGraphDatabase.driver(
            config.neo4j_url,
            auth=("neo4j", config.neo4j_password)
        )
        
    async def close(self):
        await self.driver.close()
        
    async def create_strategy_node(self, strategy_data: Dict[str, Any]) -> str:
        """Create a strategy node in the knowledge graph"""
        async with self.driver.session() as session:
            result = await session.run("""
                CREATE (s:Strategy {
                    id: $id,
                    name: $name,
                    type: $type,
                    description: $description,
                    entry_rules: $entry_rules,
                    exit_rules: $exit_rules,
                    risk_management: $risk_management,
                    timeframe: $timeframe,
                    asset_class: $asset_class,
                    created_at: datetime(),
                    source_url: $source_url,
                    source_type: $source_type
                })
                RETURN s.id as strategy_id
            """, **strategy_data)
            
            record = await result.single()
            return record["strategy_id"]
            
    async def create_indicator_node(self, indicator_data: Dict[str, Any]) -> str:
        """Create an indicator node"""
        async with self.driver.session() as session:
            result = await session.run("""
                MERGE (i:Indicator {name: $name})
                SET i.parameters = $parameters,
                    i.calculation_method = $calculation_method,
                    i.category = $category,
                    i.updated_at = datetime()
                RETURN i.name as indicator_name
            """, **indicator_data)
            
            record = await result.single()
            return record["indicator_name"]
            
    async def create_relationship(
        self,
        node1_type: str,
        node1_id: str,
        relationship_type: str,
        node2_type: str,
        node2_id: str,
        properties: Optional[Dict] = None
    ):
        """Create a relationship between nodes"""
        props = properties or {}
        
        async with self.driver.session() as session:
            await session.run(f"""
                MATCH (n1:{node1_type} {{id: $node1_id}})
                MATCH (n2:{node2_type} {{id: $node2_id}})
                CREATE (n1)-[r:{relationship_type} $properties]->(n2)
                SET r.created_at = datetime()
            """, node1_id=node1_id, node2_id=node2_id, properties=props)
            
    async def store_backtest_results(
        self,
        strategy_id: str,
        results: Dict[str, Any]
    ):
        """Store backtest results for a strategy"""
        async with self.driver.session() as session:
            await session.run("""
                MATCH (s:Strategy {id: $strategy_id})
                CREATE (r:BacktestResult {
                    strategy_id: $strategy_id,
                    total_return: $total_return,
                    sharpe_ratio: $sharpe_ratio,
                    max_drawdown: $max_drawdown,
                    win_rate: $win_rate,
                    profit_factor: $profit_factor,
                    start_date: $start_date,
                    end_date: $end_date,
                    timeframe: $timeframe,
                    initial_capital: $initial_capital,
                    final_capital: $final_capital,
                    num_trades: $num_trades,
                    avg_trade_return: $avg_trade_return,
                    volatility: $volatility,
                    sortino_ratio: $sortino_ratio,
                    calmar_ratio: $calmar_ratio,
                    created_at: datetime()
                })
                CREATE (s)-[:HAS_BACKTEST]->(r)
            """, strategy_id=strategy_id, **results)
            
    async def find_similar_strategies(
        self,
        strategy_id: str,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find strategies similar to the given strategy"""
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (s1:Strategy {id: $strategy_id})
                MATCH (s2:Strategy)
                WHERE s1 <> s2
                
                // Find common indicators
                OPTIONAL MATCH (s1)-[:USES_INDICATOR]->(i:Indicator)<-[:USES_INDICATOR]-(s2)
                WITH s1, s2, count(i) as common_indicators
                
                // Find total indicators for each strategy
                MATCH (s1)-[:USES_INDICATOR]->(i1:Indicator)
                WITH s1, s2, common_indicators, count(i1) as s1_indicators
                
                MATCH (s2)-[:USES_INDICATOR]->(i2:Indicator)
                WITH s1, s2, common_indicators, s1_indicators, count(i2) as s2_indicators
                
                // Calculate Jaccard similarity
                WITH s1, s2, common_indicators, s1_indicators, s2_indicators,
                     toFloat(common_indicators) / (s1_indicators + s2_indicators - common_indicators) as similarity
                
                WHERE similarity >= $threshold
                
                RETURN s2.id as strategy_id,
                       s2.name as name,
                       s2.type as type,
                       similarity
                ORDER BY similarity DESC
                LIMIT 10
            """, strategy_id=strategy_id, threshold=similarity_threshold)
            
            return [dict(record) for record in result]
            
    async def get_best_strategies_for_condition(
        self,
        market_condition: str,
        min_sharpe_ratio: float = 1.0
    ) -> List[Dict[str, Any]]:
        """Get best performing strategies for specific market conditions"""
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (s:Strategy)-[:HAS_BACKTEST]->(r:BacktestResult)
                MATCH (s)-[:PERFORMS_IN_CONDITION]->(c:MarketCondition {type: $condition})
                WHERE r.sharpe_ratio >= $min_sharpe
                RETURN s.id as strategy_id,
                       s.name as name,
                       s.type as type,
                       r.sharpe_ratio as sharpe_ratio,
                       r.total_return as total_return,
                       r.max_drawdown as max_drawdown
                ORDER BY r.sharpe_ratio DESC
                LIMIT 20
            """, condition=market_condition, min_sharpe=min_sharpe_ratio)
            
            return [dict(record) for record in result]
            
    async def analyze_strategy_evolution(self, base_strategy_id: str) -> Dict[str, Any]:
        """Analyze how a strategy has evolved through optimizations"""
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH path = (base:Strategy {id: $base_id})-[:OPTIMIZED_VERSION_OF*]->(original:Strategy)
                WHERE NOT (original)-[:OPTIMIZED_VERSION_OF]->()
                
                WITH base, original, length(path) as evolution_depth
                
                MATCH (base)-[:HAS_BACKTEST]->(base_result:BacktestResult)
                MATCH (original)-[:HAS_BACKTEST]->(original_result:BacktestResult)
                
                RETURN original.name as original_name,
                       base.name as current_name,
                       evolution_depth,
                       original_result.sharpe_ratio as original_sharpe,
                       base_result.sharpe_ratio as current_sharpe,
                       base_result.sharpe_ratio - original_result.sharpe_ratio as sharpe_improvement,
                       original_result.max_drawdown as original_drawdown,
                       base_result.max_drawdown as current_drawdown
            """, base_id=base_strategy_id)
            
            record = await result.single()
            return dict(record) if record else {}
```

---

## 7. Phase 4: ML/AI Integration (Week 7-8)

### 7.1 Machine Learning Pipeline

```python
# src/ml/ml_pipeline.py
import mlflow
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import joblib
from pathlib import Path

class MLPipeline:
    """Machine Learning pipeline for trading strategy enhancement"""
    
    def __init__(self):
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment("trading_strategies")
        self.models = {}
        self.feature_importance = {}
        
    async def prepare_features(
        self,
        price_data: pd.DataFrame,
        indicators: List[str] = None,
        lookback_window: int = 30
    ) -> pd.DataFrame:
        """Prepare features for ML models"""
        df = price_data.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Technical indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            
        # Forward-looking target (for supervised learning)
        df['future_returns'] = df['returns'].shift(-1)  # Next period return
        df['target_direction'] = (df['future_returns'] > 0).astype(int)
        
        return df.dropna()
        
    async def train_price_predictor(
        self,
        features_df: pd.DataFrame,
        model_type: str = "random_forest"
    ) -> Dict[str, Any]:
        """Train a price prediction model"""
        
        with mlflow.start_run(run_name=f"price_predictor_{model_type}"):
            # Prepare data
            feature_cols = [col for col in features_df.columns 
                          if col not in ['future_returns', 'target_direction', 'timestamp']]
            X = features_df[feature_cols]
            y = features_df['future_returns']
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=5)
            
            if model_type == "random_forest":
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            # Cross-validation
            cv_scores = []
            feature_importance_list = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                
                cv_scores.append({'mse': mse, 'mae': mae})
                feature_importance_list.append(
                    dict(zip(feature_cols, model.feature_importances_))
                )
                
            # Calculate average performance
            avg_mse = np.mean([score['mse'] for score in cv_scores])
            avg_mae = np.mean([score['mae'] for score in cv_scores])
            
            # Average feature importance
            avg_importance = {}
            for feature in feature_cols:
                avg_importance[feature] = np.mean([
                    fi[feature] for fi in feature_importance_list
                ])
                
            # Train final model on all data
            final_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            final_model.fit(X, y)
            
            # Log to MLflow
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_metric("cv_mse", avg_mse)
            mlflow.log_metric("cv_mae", avg_mae)
            
            # Save model
            model_path = f"models/price_predictor_{model_type}.joblib"
            Path("models").mkdir(exist_ok=True)
            joblib.dump(final_model, model_path)
            mlflow.log_artifact(model_path)
            
            # Store in class
            self.models[f"price_predictor_{model_type}"] = final_model
            self.feature_importance[f"price_predictor_{model_type}"] = avg_importance
            
            return {
                'model_type': model_type,
                'cv_mse': avg_mse,
                'cv_mae': avg_mae,
                'feature_importance': avg_importance,
                'model_path': model_path
            }
            
    async def optimize_model_hyperparameters(
        self,
        features_df: pd.DataFrame,
        model_type: str = "random_forest",
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """Optimize model hyperparameters using Optuna"""
        
        def objective(trial):
            # Suggest hyperparameters
            if model_type == "random_forest":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
                model = RandomForestRegressor(**params, random_state=42)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            # Prepare data
            feature_cols = [col for col in features_df.columns 
                          if col not in ['future_returns', 'target_direction', 'timestamp']]
            X = features_df[feature_cols]
            y = features_df['future_returns']
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)  # Reduced for faster optimization
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                scores.append(mse)
                
            return np.mean(scores)
            
        # Create study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Log best parameters to MLflow
        with mlflow.start_run(run_name=f"hyperopt_{model_type}"):
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_mse", study.best_value)
            
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': n_trials
        }
```

### 7.2 Feature Engineering Pipeline

```python
# src/ml/feature_engineering.py
import pandas as pd
import numpy as np
from tsfresh import extract_features, extract_relevant_features
from tsfresh.feature_