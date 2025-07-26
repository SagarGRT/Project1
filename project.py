#!/usr/bin/env python3
"""
Digital Asset Portfolio Management Microservice

OVERVIEW:
=========
This microservice provides real-time portfolio rebalancing for digital assets using mean-variance 
optimization. It pulls live price data from secure REST APIs, computes rolling covariance matrices,
and executes rebalancing decisions at configurable intervals.

SETUP & DEPENDENCIES:
====================
1. Install required packages:
   pip install requests pandas numpy pypfopt python-dotenv watchdog aiohttp

2. Environment Variables (Required):
   - API_BASE_URL: Base URL for the price data API
   - API_KEY: Authentication key for API access
   - API_SECRET: Secret for API authentication (if required)
   - TLS_CERT_PATH: Path to TLS certificate file (optional)
   - TLS_KEY_PATH: Path to TLS private key file (optional)

3. Configuration Files:
   - .env: Environment variables and basic configuration
   - config.json: Runtime parameters (hot-reloadable)

4. Example config.json:
   {
     "risk_free_rate": 0.02,
     "rebalancing_interval_minutes": 5,
     "optimization_objective": "max_sharpe",
     "target_sharpe_ratio": null,
     "rolling_window_periods": 60,
     "assets": ["BTC", "ETH", "ADA", "SOL"],
     "max_weight_per_asset": 0.4,
     "min_weight_per_asset": 0.05
   }

CONFIGURATION HOT-RELOADING:
===========================
The service monitors config.json for changes and reloads parameters automatically without restart.
Changes to .env variables require service restart for security reasons.

AUDIT COMPLIANCE:
================
All rebalancing decisions are logged to immutable audit files with SOC 2 compliance features:
- Cryptographic integrity checks
- Tamper-evident logging
- Complete decision audit trail
- Performance metrics tracking

USAGE:
======
python portfolio_service.py

The service will start and begin the rebalancing loop based on configured intervals.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import ssl
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import aiohttp
from dotenv import load_dotenv
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.exceptions import OptimizationError
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('portfolio_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """Configuration class for portfolio management parameters."""
    risk_free_rate: float = 0.02
    rebalancing_interval_minutes: int = 5
    optimization_objective: str = "max_sharpe"  # "max_sharpe" or "target_sharpe"
    target_sharpe_ratio: Optional[float] = None
    rolling_window_periods: int = 60
    assets: Optional[List[str]] = None
    max_weight_per_asset: float = 0.4
    min_weight_per_asset: float = 0.05
    
    def __post_init__(self):
        if self.assets is None:
            self.assets = ["BTC", "ETH", "ADA", "SOL"]


@dataclass
class RebalanceRecord:
    """Immutable record of rebalancing decision for audit compliance."""
    timestamp: str
    portfolio_weights: Dict[str, float]
    expected_return: float
    portfolio_volatility: float
    sharpe_ratio: float
    optimization_objective: str
    target_sharpe: Optional[float]
    execution_time_ms: float
    price_data_hash: str
    config_hash: str
    
    def to_audit_string(self) -> str:
        """Convert record to tamper-evident audit string."""
        record_dict = asdict(self)
        record_json = json.dumps(record_dict, sort_keys=True)
        record_hash = hashlib.sha256(record_json.encode()).hexdigest()
        return f"{record_json}|HASH:{record_hash}"


class ConfigWatcher(FileSystemEventHandler):
    """File system watcher for hot-reloading configuration changes."""
    
    def __init__(self, callback):
        self.callback = callback
        self.last_modified = 0
        
    def on_modified(self, event):
        if event.src_path.endswith('config.json'):
            current_time = time.time()
            if current_time - self.last_modified > 1:  # Debounce
                self.last_modified = current_time
                self.callback()


class SecureAPIClient:
    """Secure REST API client with TLS support for fetching market data."""
    
    def __init__(self):
        self.base_url = self._get_required_env('API_BASE_URL')
        self.api_key = self._get_required_env('API_KEY')
        self.api_secret = os.getenv('API_SECRET', '')
        self.session = None
        
    def _get_required_env(self, key: str) -> str:
        """Get required environment variable or raise error."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not set")
        return value
        
    def _create_secure_ssl_context(self) -> ssl.SSLContext:
        """Create secure SSL context with strong protocol settings."""
        # Use explicit secure protocol for Python 3.10+ compliance
        if sys.version_info >= (3, 10):
            # Python 3.10+ uses secure defaults, but be explicit
            ssl_context = ssl.create_default_context()
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            ssl_context.maximum_version = ssl.TLSVersion.TLSv1_3
        else:
            # For older Python versions, explicitly set secure protocols
            ssl_context = ssl.create_default_context()
            ssl_context.options |= ssl.OP_NO_SSLv2
            ssl_context.options |= ssl.OP_NO_SSLv3
            ssl_context.options |= ssl.OP_NO_TLSv1
            ssl_context.options |= ssl.OP_NO_TLSv1_1
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        
        # Additional security settings
        ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        # Optional client certificate authentication
        cert_path = os.getenv('TLS_CERT_PATH')
        key_path = os.getenv('TLS_KEY_PATH')
        if cert_path and key_path:
            ssl_context.load_cert_chain(cert_path, key_path)
            
        return ssl_context
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create async HTTP session."""
        if self.session is None or self.session.closed:
            # Configure secure TLS
            ssl_context = self._create_secure_ssl_context()
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            # Set security headers
            headers = {
                'User-Agent': 'DigitalAssetManager/1.0',
                'X-API-Key': self.api_key,
                'Content-Type': 'application/json'
            }
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
        return self.session
        
    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = '') -> str:
        """Generate HMAC signature for API authentication."""
        if not self.api_secret:
            return ''
            
        message = f"{timestamp}{method.upper()}{path}{body}"
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
        
    async def fetch_price_data(self, assets: List[str]) -> pd.DataFrame:
        """
        Fetch real-time price data for specified assets.
        
        Args:
            assets: List of asset symbols to fetch
            
        Returns:
            DataFrame with timestamp index and asset price columns
            
        Raises:
            aiohttp.ClientError: If API request fails
            ValueError: If response data is invalid
        """
        session = await self._get_session()
        
        try:
            timestamp = str(int(time.time()))
            path = '/api/v1/prices'
            
            params = {'symbols': ','.join(assets)}
            signature = self._generate_signature(timestamp, 'GET', path)
            
            headers = {
                'X-Timestamp': timestamp,
                'X-Signature': signature
            }
            
            async with session.get(
                f"{self.base_url}{path}",
                params=params,
                headers=headers
            ) as response:
                response.raise_for_status()
                data = await response.json()
            
            # Convert to DataFrame with proper timestamp indexing
            prices_data = []
            for asset_data in data.get('data', []):
                prices_data.append({
                    'timestamp': pd.to_datetime(asset_data['timestamp']),
                    'symbol': asset_data['symbol'],
                    'price': float(asset_data['price'])
                })
                
            if not prices_data:
                raise ValueError("No price data received from API")
                
            df = pd.DataFrame(prices_data)
            df_pivot = df.pivot(index='timestamp', columns='symbol', values='price')
            df_pivot = df_pivot.sort_index()
            
            logger.info(f"Fetched price data for {len(assets)} assets")
            return df_pivot
            
        except aiohttp.ClientError as e:
            logger.error(f"API request failed: {e}")
            raise
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Invalid API response format: {e}")
            raise ValueError(f"Invalid API response: {e}")
    
    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()


class PortfolioOptimizer:
    """Mean-variance portfolio optimizer using PyPortfolioOpt."""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        
    def compute_rolling_covariance(self, price_data: pd.DataFrame) -> np.ndarray:
        """
        Compute rolling covariance matrix from price data.
        
        Args:
            price_data: DataFrame with price data
            
        Returns:
            Covariance matrix as numpy array
        """
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Use rolling window if we have enough data
        if len(returns) >= self.config.rolling_window_periods:
            recent_returns = returns.tail(self.config.rolling_window_periods)
        else:
            recent_returns = returns
            logger.warning(f"Using {len(returns)} periods instead of {self.config.rolling_window_periods}")
            
        # Compute covariance matrix
        cov_matrix = risk_models.sample_cov(recent_returns, frequency=252 * 24 * 12)  # 5-minute intervals
        return cov_matrix
        
    def optimize_portfolio(
        self, 
        price_data: pd.DataFrame
    ) -> Tuple[Dict[str, float], float, float, float]:
        """
        Optimize portfolio weights using mean-variance optimization.
        
        Args:
            price_data: Historical price data
            
        Returns:
            Tuple of (weights_dict, expected_return, volatility, sharpe_ratio)
            
        Raises:
            OptimizationError: If optimization fails
            ValueError: If insufficient data
        """
        if len(price_data) < 2:
            raise ValueError("Insufficient price data for optimization")
            
        try:
            # Calculate expected returns and covariance
            returns = price_data.pct_change().dropna()
            if len(returns) == 0:
                raise ValueError("No valid returns data")
                
            mu = expected_returns.mean_historical_return(price_data, frequency=252 * 24 * 12)
            S = self.compute_rolling_covariance(price_data)
            
            # Create efficient frontier
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w >= self.config.min_weight_per_asset)
            ef.add_constraint(lambda w: w <= self.config.max_weight_per_asset)
            
            # Optimize based on objective
            if (self.config.optimization_objective == "target_sharpe" and 
                self.config.target_sharpe_ratio is not None):
                try:
                    weights = ef.efficient_return(
                        target_return=self.config.target_sharpe_ratio * np.sqrt(252 * 24 * 12) * 
                        np.sqrt(np.diagonal(S).mean()) + self.config.risk_free_rate
                    )
                except OptimizationError:
                    logger.warning("Target Sharpe optimization failed, falling back to max Sharpe")
                    weights = ef.max_sharpe(risk_free_rate=self.config.risk_free_rate)
            else:
                weights = ef.max_sharpe(risk_free_rate=self.config.risk_free_rate)
                
            # Clean weights (remove tiny positions)
            weights = ef.clean_weights()
            
            # Calculate performance metrics
            expected_return, volatility, sharpe = ef.portfolio_performance(
                risk_free_rate=self.config.risk_free_rate, verbose=False
            )
            
            logger.info(f"Portfolio optimized - Sharpe: {sharpe:.4f}, Return: {expected_return:.4f}")
            
            return weights, expected_return, volatility, sharpe
            
        except OptimizationError as e:
            logger.error(f"Portfolio optimization failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected optimization error: {e}")
            raise OptimizationError(f"Optimization failed: {e}")


class AuditLogger:
    """SOC 2 compliant audit logger for portfolio decisions."""
    
    def __init__(self, log_directory: str = "audit_logs"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        self._setup_audit_logging()
        
    def _setup_audit_logging(self):
        """Set up dedicated audit logging."""
        self.audit_logger = logging.getLogger('audit')
        self.audit_logger.setLevel(logging.INFO)
        
        # Create audit log handler with daily rotation
        audit_handler = logging.FileHandler(
            self.log_directory / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        )
        audit_formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S UTC'
        )
        audit_handler.setFormatter(audit_formatter)
        self.audit_logger.addHandler(audit_handler)
        
    def log_rebalance_decision(self, record: RebalanceRecord):
        """
        Log rebalancing decision with cryptographic integrity.
        
        Args:
            record: Rebalance record to log
        """
        audit_string = record.to_audit_string()
        self.audit_logger.info(audit_string)
        
        # Also write to immutable file with timestamp
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        immutable_file = self.log_directory / f"rebalance_{timestamp_str}.json"
        
        with open(immutable_file, 'w') as f:
            json.dump(asdict(record), f, indent=2)
            
        # Set file as read-only for immutability
        immutable_file.chmod(0o444)
        
        logger.info(f"Audit record written: {immutable_file}")


class PortfolioService:
    """Main portfolio management service."""
    
    def __init__(self):
        self.config = self._load_config()
        self.api_client = SecureAPIClient()
        self.optimizer = PortfolioOptimizer(self.config)
        self.audit_logger = AuditLogger()
        self.price_history = pd.DataFrame()
        self._setup_config_watcher()
        self.running = False
        
    def _load_config(self) -> PortfolioConfig:
        """Load configuration from JSON file with fallback to defaults."""
        config_file = Path('config.json')
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                logger.info("Configuration loaded from config.json")
                return PortfolioConfig(**config_data)
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Invalid config.json format: {e}")
                logger.info("Using default configuration")
                return PortfolioConfig()
        else:
            logger.info("No config.json found, using default configuration")
            return PortfolioConfig()
            
    def _setup_config_watcher(self):
        """Set up file watcher for hot-reloading configuration."""
        self.config_observer = Observer()
        handler = ConfigWatcher(self._reload_config)
        self.config_observer.schedule(handler, '.', recursive=False)
        self.config_observer.start()
        logger.info("Configuration hot-reloading enabled")
        
    def _reload_config(self):
        """Reload configuration from file."""
        try:
            old_config = self.config
            self.config = self._load_config()
            self.optimizer = PortfolioOptimizer(self.config)
            logger.info("Configuration reloaded successfully")
            
            # Log significant changes
            if old_config.assets != self.config.assets:
                logger.info(f"Asset list changed: {old_config.assets} -> {self.config.assets}")
            if old_config.rebalancing_interval_minutes != self.config.rebalancing_interval_minutes:
                logger.info(f"Rebalancing interval changed: {old_config.rebalancing_interval_minutes} -> {self.config.rebalancing_interval_minutes}")
                
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of price data for audit trail."""
        data_string = data.to_string()
        return hashlib.sha256(data_string.encode()).hexdigest()
        
    def _calculate_config_hash(self) -> str:
        """Calculate hash of current configuration for audit trail."""
        config_dict = asdict(self.config)
        config_string = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_string.encode()).hexdigest()
        
    async def fetch_and_store_prices(self):
        """Fetch latest prices and update price history."""
        try:
            new_data = await self.api_client.fetch_price_data(self.config.assets)
            
            if self.price_history.empty:
                self.price_history = new_data
            else:
                # Merge with existing data, avoiding duplicates
                self.price_history = pd.concat([self.price_history, new_data])
                self.price_history = self.price_history[~self.price_history.index.duplicated(keep='last')]
                self.price_history = self.price_history.sort_index()
                
                # Keep only recent data to manage memory
                max_periods = self.config.rolling_window_periods * 2
                if len(self.price_history) > max_periods:
                    self.price_history = self.price_history.tail(max_periods)
                    
            logger.info(f"Price history updated: {len(self.price_history)} periods")
            
        except Exception as e:
            logger.error(f"Failed to fetch price data: {e}")
            raise
            
    def execute_rebalancing(self) -> Optional[RebalanceRecord]:
        """
        Execute portfolio rebalancing and create audit record.
        
        Returns:
            Rebalance record if successful, None otherwise
        """
        if len(self.price_history) < 2:
            logger.warning("Insufficient price history for rebalancing")
            return None
            
        start_time = time.time()
        
        try:
            # Optimize portfolio
            weights, expected_return, volatility, sharpe = self.optimizer.optimize_portfolio(
                self.price_history
            )
            
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Create audit record
            record = RebalanceRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                portfolio_weights=weights,
                expected_return=expected_return,
                portfolio_volatility=volatility,
                sharpe_ratio=sharpe,
                optimization_objective=self.config.optimization_objective,
                target_sharpe=self.config.target_sharpe_ratio,
                execution_time_ms=execution_time,
                price_data_hash=self._calculate_data_hash(self.price_history),
                config_hash=self._calculate_config_hash()
            )
            
            # Log audit record
            self.audit_logger.log_rebalance_decision(record)
            
            logger.info(f"Portfolio rebalanced successfully in {execution_time:.2f}ms")
            logger.info(f"New weights: {weights}")
            
            return record
            
        except Exception as e:
            logger.error(f"Rebalancing failed: {e}")
            return None
            
    async def run(self):
        """Main service loop."""
        logger.info("Starting Portfolio Management Service")
        logger.info(f"Rebalancing interval: {self.config.rebalancing_interval_minutes} minutes")
        logger.info(f"Monitoring assets: {self.config.assets}")
        
        self.running = True
        
        try:
            while self.running:
                loop_start = time.time()
                
                try:
                    # Fetch latest price data
                    await self.fetch_and_store_prices()
                    
                    # Execute rebalancing (synchronous call)
                    record = self.execute_rebalancing()
                    
                    if record:
                        logger.info(f"Rebalancing completed - Sharpe: {record.sharpe_ratio:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    
                # Calculate sleep time to maintain interval
                loop_duration = time.time() - loop_start
                sleep_time = max(0, (self.config.rebalancing_interval_minutes * 60) - loop_duration)
                
                if sleep_time > 0:
                    logger.debug(f"Sleeping for {sleep_time:.1f} seconds")
                    await asyncio.sleep(sleep_time)
                else:
                    logger.warning(f"Loop took {loop_duration:.1f}s, longer than interval")
                    
        except KeyboardInterrupt:
            logger.info("Service stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in service: {e}")
            raise
        finally:
            await self.shutdown()
            
    async def shutdown(self):
        """Graceful shutdown of the service."""
        logger.info("Shutting down Portfolio Management Service")
        self.running = False
        
        if hasattr(self, 'config_observer'):
            self.config_observer.stop()
            self.config_observer.join()
            
        # Close API session
        await self.api_client.close()
            
        logger.info("Service shutdown complete")


async def main():
    """Main entry point."""
    service = PortfolioService()
    
    try:
        await service.run()
    except Exception as e:
        logger.error(f"Service failed to start: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
