# Path management utilities

from pathlib import Path
from typing import Union, List


class ProjectPaths:
    """Centralized path management for the project"""

    def __init__(self, project_root: Union[str, Path] = None):
        if project_root is None:
            # Auto-detect project root (assuming this file is in src/utils/)
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)

    @property
    def data_raw(self) -> Path:
        """Raw data directory"""
        return self.project_root / 'data' / 'raw'

    @property
    def data_processed(self) -> Path:
        """Processed data directory"""
        return self.project_root / 'data' / 'processed'

    @property
    def data_external(self) -> Path:
        """External data directory"""
        return self.project_root / 'data' / 'external'

    @property
    def features(self) -> Path:
        """Features directory"""
        return self.project_root / 'features'

    @property
    def results(self) -> Path:
        """Results directory"""
        return self.project_root / 'results'

    @property
    def experiments(self) -> Path:
        """Experiments directory"""
        return self.project_root / 'experiments'

    @property
    def scripts(self) -> Path:
        """Scripts directory"""
        return self.project_root / 'scripts'

    def get_member_results_path(self, member: str) -> Path:
        """Get results path for a specific member"""
        model_names = {
            'A': '1dcnn',
            'B': '2dcnn', 
            'C': 'transformer',
            'D': 'multimodal',
            'E': 'baseline'
        }
        model_name = model_names.get(member, member.lower())
        return self.results / f'member{member}_{model_name}'

    def get_member_experiments_path(self, member: str) -> Path:
        """Get experiments path for a specific member"""
        model_names = {
            'A': '1dcnn',
            'B': '2dcnn',
            'C': 'transformer', 
            'D': 'multimodal',
            'E': 'baseline'
        }
        model_name = model_names.get(member, member.lower())
        return self.experiments / f'member{member}_{model_name}'

    def ensure_dir(self, path: Union[str, Path]) -> Path:
        """Ensure directory exists and return Path object"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_config_path(self, filename: str = 'config.yaml') -> Path:
        """Get config file path"""
        return self.project_root / filename


# Global instance for easy access
paths = ProjectPaths()