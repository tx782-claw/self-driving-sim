"""
YAML 场景配置加载器 (v0.2.2 优化D)
=====================================
让 scenarios / sensors / tracker 参数可以从 yaml 文件加载，
方便批量 sweep 实验和 CI 集成。

用法:
    from scenarios.yaml_loader import load_scenarios_from_yaml
    configs = load_scenarios_from_yaml("config/scenarios.yaml")
    for cfg in configs:
        scenario = cfg['scenario']
        sensors = cfg['sensors']
        tracker_params = cfg['tracker_params']
        # ... 跑仿真
"""
import os
import yaml
import numpy as np
from sensors import LidarSensor, RadarSensor, CameraSensor, IMUSensor, GPSSensor


_SENSOR_REGISTRY = {
    'LidarSensor': LidarSensor,
    'RadarSensor': RadarSensor,
    'CameraSensor': CameraSensor,
    'IMUSensor': IMUSensor,
    'GPSSensor': GPSSensor,
}


def _npify(d):
    """递归把 list → np.array（顶层值是 list 的）"""
    if isinstance(d, dict):
        return {k: _npify(v) for k, v in d.items()}
    if isinstance(d, list) and d and all(isinstance(x, (int, float)) for x in d):
        return np.array(d)
    return d


def _build_scenario(stype: str, params: dict):
    """从 type + params 构造场景对象"""
    from scenarios import HighwayScenario, DenseHighwayScenario, JunctionScenario, StopAndGoScenario, UrbanScenario
    registry = {
        'highway': HighwayScenario,
        'dense': DenseHighwayScenario,
        'junction': JunctionScenario,
        'stop_and_go': StopAndGoScenario,
        'urban': UrbanScenario,
    }
    if stype not in registry:
        raise ValueError(f"未知场景类型: {stype} (支持: {list(registry.keys())})")
    return registry[stype](**_npify(params))


def _build_sensors(sensor_specs: dict) -> dict:
    """从 spec dict 构造传感器实例"""
    sensors = {}
    for sid, spec in sensor_specs.items():
        cls_name = spec['class']
        if cls_name not in _SENSOR_REGISTRY:
            raise ValueError(f"未知传感器类: {cls_name}")
        sensors[sid] = _SENSOR_REGISTRY[cls_name](sensor_id=sid,
                                                   **_npify(spec['params']))
    return sensors


def load_scenarios_from_yaml(yaml_path: str) -> list[dict]:
    """
    从 yaml 文件加载所有场景配置

    Returns:
        list of {name, type, scenario, sensors, tracker_params}
    """
    if not os.path.isabs(yaml_path):
        # 相对路径：相对项目根
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        yaml_path = os.path.join(project_root, yaml_path)

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"yaml 不存在: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    default_sensors = cfg.get('default_sensors', {})
    default_tracker = cfg.get('default_tracker', {}).get('params', {})

    result = []
    for sc in cfg.get('scenarios', []):
        name = sc['name']
        stype = sc['type']
        params = sc['params']
        # 单场景 sensor 覆盖默认（可选，目前未在 yaml 里实现）
        sensors_spec = default_sensors
        result.append({
            'name': name,
            'type': stype,
            'scenario': _build_scenario(stype, params),
            'sensors': _build_sensors(sensors_spec),
            'tracker_params': default_tracker.copy(),
        })
    return result


def list_scenario_names(yaml_path: str) -> list[str]:
    """只读名字（不构造对象）"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(yaml_path):
        yaml_path = os.path.join(project_root, yaml_path)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return [s['name'] for s in cfg.get('scenarios', [])]
