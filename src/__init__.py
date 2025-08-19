"""Cross-country q# Try to import models with fallback
try:
    from .factor_nn import FactorNeuralNetwork, EnsembleFactorNN
    from .factor_nn_api import FactorNN
except ImportError:
    FactorNeuralNetwork = None
    EnsembleFactorNN = None
    FactorNN = None

try:
    from .lqr import LinearQuantileRegression
    from .lqr_api import LQRModel, LQR
except ImportError:
    LinearQuantileRegression = None
    LQRModel = None
    LQR = Nonecasting package."""

# Import available classes and functions
from .utils import (
    load_country_data,
    load_config,
    set_seeds
)
from .metrics import (
    pinball_loss,
    compute_quantile_losses,
    diebold_mariano_test
)
from .evaluation import (
    create_model_comparison_table,
    create_evaluation_dashboard
)

# Try to import models with fallback
try:
    from .ensembleNN import  EnsembleNN
except ImportError:
    EnsembleNN = None

try:
    from .lqr import LinearQuantileRegression, LQRModel
except ImportError:
    LinearQuantileRegression = None
    LQRModel = None

__version__ = "0.1.0"

__all__ = [
    "load_country_data",
    "load_config", 
    "set_seeds",
    "pinball_loss",
    "compute_quantile_losses",
    "diebold_mariano_test",
    "create_model_comparison_table",
    "create_evaluation_dashboard",
    "EnsembleNN",    
    "LinearQuantileRegression",
    "LQRModel",
    "LQR"
]
