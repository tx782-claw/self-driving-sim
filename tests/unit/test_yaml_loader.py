"""
YAML 场景加载器单测
"""
import os
import pytest
from scenarios.yaml_loader import load_scenarios_from_yaml, list_scenario_names


class TestYAMLLoader:
    """YAML 场景配置加载"""

    def test_list_names(self):
        names = list_scenario_names('config/scenarios.yaml')
        assert 'highway_baseline_5v' in names
        assert 'dense_24v' in names
        assert len(names) >= 3

    def test_load_scenarios(self):
        configs = load_scenarios_from_yaml('config/scenarios.yaml')
        assert len(configs) >= 3
        for c in configs:
            assert 'name' in c
            assert 'scenario' in c
            assert 'sensors' in c
            assert 'tracker_params' in c
            # scenario 应该是 Highway/Dense/StopAndGo 之一
            assert type(c['scenario']).__name__ in (
                'HighwayScenario', 'DenseHighwayScenario',
                'StopAndGoScenario', 'JunctionScenario', 'UrbanScenario')

    def test_sensors_built_correctly(self):
        configs = load_scenarios_from_yaml('config/scenarios.yaml')
        for c in configs:
            sensors = c['sensors']
            assert 'lidar_top' in sensors
            assert 'radar_front' in sensors
            assert 'imu' in sensors
            assert 'gps' in sensors
            # LiDAR max_range 应该是 80
            assert sensors['lidar_top'].max_range == 80.0

    def test_tracker_params_have_keys(self):
        configs = load_scenarios_from_yaml('config/scenarios.yaml')
        params = configs[0]['tracker_params']
        assert params['gate_threshold'] == 15.0
        assert params['association_mode'] == 'hungarian'

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_scenarios_from_yaml('config/nonexistent.yaml')
