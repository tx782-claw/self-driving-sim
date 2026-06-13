"""
测试配置 — 让 pytest 能 import 项目模块
"""
import sys
import os

# 项目根目录加入 path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
