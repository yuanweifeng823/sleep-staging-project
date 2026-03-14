# Configuration management utilities

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from ..utils.logger import setup_logger
from ..utils.paths import paths

logger = setup_logger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    scheduler: str = 'cosine'
    grad_clip: Optional[float] = 1.0
    patience: int = 10
    save_freq: int = 10

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelConfig:
    """Model configuration dataclass"""
    n_classes: int = 5
    input_channels: int = 1
    hidden_dims: list = None
    dropout: float = 0.5
    activation: str = 'relu'

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 256]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DataConfig:
    """Data configuration dataclass"""
    sampling_rate: int = 100
    epoch_length: int = 30
    overlap: float = 0.0
    test_fold: int = 0
    n_folds: int = 5
    modalities: list = None

    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ['eeg']

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    name: str = 'default_experiment'
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    seed: int = 42
    device: str = 'auto'
    log_level: str = 'INFO'

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from nested dictionary"""
        config = cls()

        # Handle nested configs
        if 'model' in data:
            config.model = ModelConfig.from_dict(data['model'])
        if 'training' in data:
            config.training = TrainingConfig.from_dict(data['training'])
        if 'data' in data:
            config.data = DataConfig.from_dict(data['data'])

        # Handle top-level fields
        for key, value in data.items():
            if key not in ['model', 'training', 'data'] and hasattr(config, key):
                setattr(config, key, value)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        # Convert nested dataclasses to dicts
        result['model'] = asdict(self.model)
        result['training'] = asdict(self.training)
        result['data'] = asdict(self.data)
        return result


class ConfigManager:
    """Configuration manager for loading and saving configs"""

    def __init__(self):
        self.configs_cache = {}

    def load_config(self, config_path: Union[str, Path]) -> ExperimentConfig:
        """Load configuration from YAML file"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Check cache
        cache_key = str(config_path.resolve())
        if cache_key in self.configs_cache:
            return self.configs_cache[cache_key]

        logger.info(f"Loading config from {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        config = ExperimentConfig.from_dict(data)

        # Cache the config
        self.configs_cache[cache_key] = config

        return config

    def save_config(self, config: ExperimentConfig, config_path: Union[str, Path]):
        """Save configuration to YAML file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = config.to_dict()

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved config to {config_path}")

    def get_default_config_path(self, member: str, config_type: str = 'train') -> Path:
        """Get default config path for a member"""
        return paths.get_member_experiments_path(member) / f'{config_type}_config.yaml'

    def create_default_configs(self):
        """Create default configuration files for all members"""
        members = ['A', 'B', 'C', 'D', 'E']

        for member in members:
            exp_path = paths.get_member_experiments_path(member)
            exp_path.mkdir(parents=True, exist_ok=True)

            # Create default training config
            train_config = ExperimentConfig(name=f'member{member}_experiment')
            config_path = self.get_default_config_path(member, 'train')
            self.save_config(train_config, config_path)

            logger.info(f"Created default config for member {member}: {config_path}")

    def validate_config(self, config: ExperimentConfig) -> bool:
        """Validate configuration"""
        errors = []

        # Validate training config
        if config.training.epochs <= 0:
            errors.append("epochs must be positive")
        if config.training.batch_size <= 0:
            errors.append("batch_size must be positive")
        if config.training.learning_rate <= 0:
            errors.append("learning_rate must be positive")

        # Validate model config
        if config.model.n_classes <= 0:
            errors.append("n_classes must be positive")
        if config.model.input_channels <= 0:
            errors.append("input_channels must be positive")

        # Validate data config
        if config.data.sampling_rate <= 0:
            errors.append("sampling_rate must be positive")
        if config.data.epoch_length <= 0:
            errors.append("epoch_length must be positive")

        if errors:
            logger.error(f"Configuration validation failed: {', '.join(errors)}")
            return False

        return True


# Global instance
config_manager = ConfigManager()