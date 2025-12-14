# Type stubs for hft_py module
from typing import Any

# Re-export all submodules
from . import statistical_analyzer as statistical_analyzer
from . import superspace as superspace
from . import signature as signature
from . import rough_heston as rough_heston
from . import flat_file as flat_file
from . import analytics as analytics
from . import options as options
from . import portfolio_drift as portfolio_drift

class PyAggregator:
    def __init__(self) -> None: ...
    def subscribe(self, connector: str, symbol: str) -> None: ...
    def unsubscribe(self, connector: str, symbol: str) -> None: ...
    def stop_connector(self, connector: str) -> bool: ...

__all__ = [
    'PyAggregator',
    'statistical_analyzer',
    'superspace',
    'signature',
    'rough_heston',
    'flat_file',
    'analytics',
    'options',
    'portfolio_drift',
]
