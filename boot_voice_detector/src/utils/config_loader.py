import yaml
import json
from pathlib import Path
import os
from typing import Dict, Any, Optional
class ConfigLoader:
    """Класс для загрузки конфигурационных файлов"""
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")
        if config_path.suffix in ['.yaml', '.yml']:
            return ConfigLoader._load_yaml(config_path)
        elif config_path.suffix == '.json':
            return ConfigLoader._load_json(config_path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {config_path.suffix}")
    @staticmethod
    def _load_yaml(config_path: Path) -> Dict[str, Any]:
        """Загрузка YAML файла"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    @staticmethod
    def _load_json(config_path: Path) -> Dict[str, Any]:
        """Загрузка JSON файла"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        if config_path.suffix in ['.yaml', '.yml']:
            ConfigLoader._save_yaml(config, config_path)
        elif config_path.suffix == '.json':
            ConfigLoader._save_json(config, config_path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {config_path.suffix}")
    @staticmethod
    def _save_yaml(config: Dict[str, Any], config_path: Path) -> None:
        """Сохранение в YAML файл"""
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    @staticmethod
    def _save_json(config: Dict[str, Any], config_path: Path) -> None:
        """Сохранение в JSON файл"""
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        result = base_config.copy()
        for key, value in override_config.items():
            if (key in result and isinstance(result[key], dict) 
                and isinstance(value, dict)):
                result[key] = ConfigLoader.merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    @staticmethod
    def load_config_with_overrides(base_config_path: str, 
                                 override_config_path: Optional[str] = None) -> Dict[str, Any]:
        base_config = ConfigLoader.load_config(base_config_path)
        if override_config_path and Path(override_config_path).exists():
            override_config = ConfigLoader.load_config(override_config_path)
            return ConfigLoader.merge_configs(base_config, override_config)
        return base_config
def load_config(config_path: str) -> Dict[str, Any]:
    """Функция-обертка для загрузки конфигурации"""
    return ConfigLoader.load_config(config_path)
def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Функция-обертка для сохранения конфигурации"""
    ConfigLoader.save_config(config, config_path)
def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Функция-обертка для слияния конфигураций"""
    return ConfigLoader.merge_configs(base_config, override_config)
def load_config_with_overrides(base_config_path: str, 
                             override_config_path: Optional[str] = None) -> Dict[str, Any]:
    """Функция-обертка для загрузки конфигурации с переопределениями"""
    return ConfigLoader.load_config_with_overrides(base_config_path, override_config_path)