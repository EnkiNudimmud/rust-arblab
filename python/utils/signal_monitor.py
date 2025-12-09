"""
Live Signal Monitoring and Alerting System

Monitors trading signals in real-time and triggers alerts based on:
- Signal threshold crossings
- Regime changes
- Portfolio risk limits
- Significant price movements
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Trading alert"""
    timestamp: datetime
    symbol: str
    alert_type: str  # 'signal', 'regime_change', 'risk', 'price_move'
    severity: str  # 'info', 'warning', 'critical'
    message: str
    value: float
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'value': self.value,
            'metadata': self.metadata
        }


class SignalMonitor:
    """
    Monitor trading signals and trigger alerts.
    """
    
    def __init__(self, 
                 alert_file: Optional[str] = None,
                 verbose: bool = True):
        """
        Args:
            alert_file: Path to save alerts (JSON)
            verbose: Print alerts to console
        """
        self.alerts: List[Alert] = []
        self.alert_file = alert_file
        self.verbose = verbose
        self.alert_handlers: List[Callable] = []
        
        # Alert thresholds
        self.thresholds = {
            'signal_strength': 2.0,      # Z-score threshold
            'regime_prob': 0.8,           # Regime probability threshold
            'volatility_spike': 2.5,      # Vol spike threshold (std devs)
            'drawdown': 0.10,             # 10% drawdown alert
            'position_size': 0.25,        # 25% of portfolio
        }
        
        # State tracking
        self.last_regimes: Dict[str, str] = {}
        self.last_prices: Dict[str, float] = {}
        self.last_signals: Dict[str, float] = {}
        
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add custom alert handler function."""
        self.alert_handlers.append(handler)
        
    def check_signal_threshold(self, 
                               symbol: str, 
                               signal_value: float,
                               signal_type: str = "z_score") -> Optional[Alert]:
        """
        Check if signal crosses threshold.
        
        Args:
            symbol: Asset symbol
            signal_value: Signal strength
            signal_type: Type of signal
            
        Returns:
            Alert if threshold crossed, None otherwise
        """
        threshold = self.thresholds['signal_strength']
        
        if abs(signal_value) > threshold:
            direction = "BUY" if signal_value < 0 else "SELL"
            severity = "warning" if abs(signal_value) > threshold else "info"
            if abs(signal_value) > threshold * 1.5:
                severity = "critical"
            
            alert = Alert(
                timestamp=datetime.now(),
                symbol=symbol,
                alert_type='signal',
                severity=severity,
                message=f"{direction} signal triggered: {signal_type} = {signal_value:.2f}",
                value=signal_value,
                metadata={'signal_type': signal_type, 'direction': direction}
            )
            
            self._trigger_alert(alert)
            return alert
        
        return None
    
    def check_regime_change(self, 
                           symbol: str, 
                           new_regime: str,
                           regime_probs: Dict[str, float]) -> Optional[Alert]:
        """
        Check for regime change.
        
        Args:
            symbol: Asset symbol
            new_regime: New regime detected
            regime_probs: Probabilities for each regime
            
        Returns:
            Alert if regime changed, None otherwise
        """
        old_regime = self.last_regimes.get(symbol)
        
        if old_regime and old_regime != new_regime:
            # Check if high confidence in new regime
            new_prob = regime_probs.get(new_regime, 0)
            
            if new_prob > self.thresholds['regime_prob']:
                alert = Alert(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    alert_type='regime_change',
                    severity='warning',
                    message=f"Regime change: {old_regime} → {new_regime} (confidence: {new_prob:.1%})",
                    value=new_prob,
                    metadata={
                        'old_regime': old_regime,
                        'new_regime': new_regime,
                        'probabilities': regime_probs
                    }
                )
                
                self._trigger_alert(alert)
                self.last_regimes[symbol] = new_regime
                return alert
        
        self.last_regimes[symbol] = new_regime
        return None
    
    def check_volatility_spike(self,
                               symbol: str,
                               current_vol: float,
                               historical_vol_mean: float,
                               historical_vol_std: float) -> Optional[Alert]:
        """
        Check for volatility spike.
        
        Args:
            symbol: Asset symbol
            current_vol: Current volatility
            historical_vol_mean: Historical mean volatility
            historical_vol_std: Historical volatility std dev
            
        Returns:
            Alert if volatility spiked, None otherwise
        """
        if historical_vol_std > 1e-10:
            z_score = (current_vol - historical_vol_mean) / historical_vol_std
            
            if z_score > self.thresholds['volatility_spike']:
                alert = Alert(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    alert_type='volatility_spike',
                    severity='warning',
                    message=f"Volatility spike: {current_vol:.2%} ({z_score:.1f}σ above mean)",
                    value=current_vol,
                    metadata={
                        'z_score': z_score,
                        'historical_mean': historical_vol_mean,
                        'historical_std': historical_vol_std
                    }
                )
                
                self._trigger_alert(alert)
                return alert
        
        return None
    
    def check_price_movement(self,
                            symbol: str,
                            current_price: float,
                            threshold_pct: float = 0.05) -> Optional[Alert]:
        """
        Check for significant price movement.
        
        Args:
            symbol: Asset symbol
            current_price: Current price
            threshold_pct: Alert threshold (e.g., 0.05 = 5%)
            
        Returns:
            Alert if significant move, None otherwise
        """
        last_price = self.last_prices.get(symbol)
        
        if last_price and last_price > 0:
            pct_change = (current_price - last_price) / last_price
            
            if abs(pct_change) > threshold_pct:
                direction = "UP" if pct_change > 0 else "DOWN"
                severity = "warning" if abs(pct_change) > threshold_pct else "info"
                if abs(pct_change) > threshold_pct * 2:
                    severity = "critical"
                
                alert = Alert(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    alert_type='price_move',
                    severity=severity,
                    message=f"Large price move {direction}: {pct_change:+.2%}",
                    value=pct_change,
                    metadata={
                        'last_price': last_price,
                        'current_price': current_price,
                        'direction': direction
                    }
                )
                
                self._trigger_alert(alert)
        
        self.last_prices[symbol] = current_price
        return None
    
    def check_portfolio_risk(self,
                            portfolio_values: Dict[str, float],
                            total_value: float) -> List[Alert]:
        """
        Check portfolio risk metrics.
        
        Args:
            portfolio_values: {symbol: position_value}
            total_value: Total portfolio value
            
        Returns:
            List of alerts
        """
        alerts = []
        
        # Check concentration risk
        for symbol, value in portfolio_values.items():
            position_pct = abs(value) / total_value if total_value > 0 else 0
            
            if position_pct > self.thresholds['position_size']:
                alert = Alert(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    alert_type='risk',
                    severity='warning',
                    message=f"Position size alert: {position_pct:.1%} of portfolio",
                    value=position_pct,
                    metadata={'position_value': value, 'total_value': total_value}
                )
                
                self._trigger_alert(alert)
                alerts.append(alert)
        
        return alerts
    
    def _trigger_alert(self, alert: Alert):
        """Trigger an alert."""
        self.alerts.append(alert)
        
        # Console logging
        if self.verbose:
            color = {
                'info': '\033[94m',      # Blue
                'warning': '\033[93m',   # Yellow
                'critical': '\033[91m'   # Red
            }.get(alert.severity, '')
            reset = '\033[0m'
            
            print(f"{color}[{alert.severity.upper()}] {alert.timestamp.strftime('%H:%M:%S')} "
                  f"{alert.symbol}: {alert.message}{reset}")
        
        # File logging
        if self.alert_file:
            self._save_alert_to_file(alert)
        
        # Custom handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def _save_alert_to_file(self, alert: Alert):
        """Save alert to JSON file."""
        try:
            path = Path(self.alert_file)  # type: ignore[arg-type]
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Append to file
            with open(path, 'a') as f:
                f.write(json.dumps(alert.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to save alert: {e}")
    
    def get_alerts(self, 
                   since: Optional[datetime] = None,
                   symbol: Optional[str] = None,
                   alert_type: Optional[str] = None,
                   severity: Optional[str] = None) -> List[Alert]:
        """
        Get filtered alerts.
        
        Args:
            since: Only alerts after this time
            symbol: Filter by symbol
            alert_type: Filter by type
            severity: Filter by severity
            
        Returns:
            Filtered list of alerts
        """
        filtered = self.alerts
        
        if since:
            filtered = [a for a in filtered if a.timestamp >= since]
        
        if symbol:
            filtered = [a for a in filtered if a.symbol == symbol]
        
        if alert_type:
            filtered = [a for a in filtered if a.alert_type == alert_type]
        
        if severity:
            filtered = [a for a in filtered if a.severity == severity]
        
        return filtered
    
    def get_alert_summary(self, since: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get summary of alerts.
        
        Args:
            since: Only alerts after this time
            
        Returns:
            DataFrame with alert summary
        """
        alerts = self.get_alerts(since=since)
        
        if not alerts:
            return pd.DataFrame()
        
        data = [a.to_dict() for a in alerts]
        df = pd.DataFrame(data)
        
        return df
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts.clear()


# Email alert handler (requires email configuration)
def email_alert_handler(alert: Alert, 
                        smtp_server: str,
                        smtp_port: int,
                        from_addr: str,
                        to_addr: str,
                        password: str):
    """
    Send alert via email.
    
    Args:
        alert: Alert to send
        smtp_server: SMTP server address
        smtp_port: SMTP port
        from_addr: From email address
        to_addr: To email address
        password: Email password
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    try:
        msg = MIMEMultipart()
        msg['From'] = from_addr
        msg['To'] = to_addr
        msg['Subject'] = f"Trading Alert: {alert.symbol} - {alert.alert_type}"
        
        body = f"""
        Alert Details:
        
        Symbol: {alert.symbol}
        Type: {alert.alert_type}
        Severity: {alert.severity}
        Time: {alert.timestamp}
        
        Message: {alert.message}
        Value: {alert.value}
        
        Metadata: {json.dumps(alert.metadata, indent=2)}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(from_addr, password)
            server.send_message(msg)
            
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")


# Webhook alert handler
def webhook_alert_handler(alert: Alert, webhook_url: str):
    """
    Send alert to webhook (e.g., Slack, Discord, custom endpoint).
    
    Args:
        alert: Alert to send
        webhook_url: Webhook URL
    """
    import requests
    
    try:
        payload = {
            'text': f"**{alert.severity.upper()}**: {alert.symbol}",
            'attachments': [{
                'fields': [
                    {'title': 'Type', 'value': alert.alert_type, 'short': True},
                    {'title': 'Time', 'value': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), 'short': True},
                    {'title': 'Message', 'value': alert.message, 'short': False},
                    {'title': 'Value', 'value': str(alert.value), 'short': True},
                ]
            }]
        }
        
        response = requests.post(webhook_url, json=payload, timeout=5)
        response.raise_for_status()
        
    except Exception as e:
        logger.error(f"Failed to send webhook alert: {e}")


if __name__ == "__main__":
    # Test monitoring system
    monitor = SignalMonitor(verbose=True)
    
    # Test signal alert
    monitor.check_signal_threshold("AAPL", 2.5, "z_score")
    
    # Test regime change
    monitor.check_regime_change("AAPL", "trending", {
        'mean_reverting': 0.1,
        'trending': 0.85,
        'high_volatility': 0.05
    })
    
    # Test volatility spike
    monitor.check_volatility_spike("AAPL", 0.05, 0.02, 0.01)
    
    # Test price movement
    monitor.last_prices["AAPL"] = 150.0
    monitor.check_price_movement("AAPL", 157.5)
    
    # Get summary
    summary = monitor.get_alert_summary()
    print("\nAlert Summary:")
    print(summary)
