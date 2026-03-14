#!/usr/bin/env python
"""Configuration management script"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import config_manager, ExperimentConfig
from src.utils.logger import setup_logger
from src.utils.paths import paths

logger = setup_logger(__name__)


def create_default_configs():
    """Create default configuration files for all members"""
    logger.info("Creating default configuration files...")

    try:
        config_manager.create_default_configs()
        logger.info("Default configurations created successfully!")
    except Exception as e:
        logger.error(f"Failed to create default configs: {e}")
        return False

    return True


def validate_config(config_path: str):
    """Validate a configuration file"""
    logger.info(f"Validating config: {config_path}")

    try:
        config = config_manager.load_config(config_path)

        if config_manager.validate_config(config):
            logger.info("Configuration is valid!")
            return True
        else:
            logger.error("Configuration validation failed!")
            return False

    except Exception as e:
        logger.error(f"Failed to validate config: {e}")
        return False


def show_config(config_path: str):
    """Display configuration contents"""
    logger.info(f"Loading config: {config_path}")

    try:
        config = config_manager.load_config(config_path)

        print("\n" + "="*50)
        print(f"Configuration: {config_path}")
        print("="*50)

        print(f"Name: {config.name}")
        print(f"Seed: {config.seed}")
        print(f"Device: {config.device}")
        print(f"Log Level: {config.log_level}")

        print("\nModel Config:")
        print(f"  Classes: {config.model.n_classes}")
        print(f"  Input Channels: {config.model.input_channels}")
        print(f"  Hidden Dims: {config.model.hidden_dims}")
        print(f"  Dropout: {config.model.dropout}")
        print(f"  Activation: {config.model.activation}")

        print("\nTraining Config:")
        print(f"  Epochs: {config.training.epochs}")
        print(f"  Batch Size: {config.training.batch_size}")
        print(f"  Learning Rate: {config.training.learning_rate}")
        print(f"  Weight Decay: {config.training.weight_decay}")
        print(f"  Scheduler: {config.training.scheduler}")
        print(f"  Grad Clip: {config.training.grad_clip}")
        print(f"  Patience: {config.training.patience}")

        print("\nData Config:")
        print(f"  Sampling Rate: {config.data.sampling_rate}")
        print(f"  Epoch Length: {config.data.epoch_length}")
        print(f"  Overlap: {config.data.overlap}")
        print(f"  Test Fold: {config.data.test_fold}")
        print(f"  N Folds: {config.data.n_folds}")
        print(f"  Modalities: {config.data.modalities}")

        print("="*50)

    except Exception as e:
        logger.error(f"Failed to load config: {e}")


def create_custom_config(member: str, output_path: str):
    """Create a custom configuration file"""
    logger.info(f"Creating custom config for member {member}")

    try:
        # Load default config
        default_path = paths.get_member_experiments_path(member) / 'train_config.yaml'

        if default_path.exists():
            config = config_manager.load_config(default_path)
            logger.info("Loaded existing config as template")
        else:
            config = ExperimentConfig(name=f'member{member}_custom_experiment')
            logger.info("Created new config from defaults")

        # Save to custom path
        config_manager.save_config(config, output_path)
        logger.info(f"Custom config saved to: {output_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to create custom config: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Configuration management tool')
    parser.add_argument('action', choices=['create', 'validate', 'show', 'custom'],
                       help='Action to perform')
    parser.add_argument('--config', '-c', type=str,
                       help='Path to configuration file')
    parser.add_argument('--member', '-m', type=str, choices=['A', 'B', 'C', 'D', 'E'],
                       help='Member identifier (for custom config)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output path (for custom config)')

    args = parser.parse_args()

    if args.action == 'create':
        success = create_default_configs()

    elif args.action == 'validate':
        if not args.config:
            logger.error("Please specify config path with --config")
            return
        success = validate_config(args.config)

    elif args.action == 'show':
        if not args.config:
            logger.error("Please specify config path with --config")
            return
        show_config(args.config)
        return  # show_config handles its own success/failure

    elif args.action == 'custom':
        if not args.member:
            logger.error("Please specify member with --member")
            return
        if not args.output:
            logger.error("Please specify output path with --output")
            return
        success = create_custom_config(args.member, args.output)

    if 'success' in locals() and not success:
        sys.exit(1)


if __name__ == "__main__":
    main()