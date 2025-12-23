#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练脚本，调用训练模块进行模型训练
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.full_training_evaluation import main as training_main

if __name__ == "__main__":
    training_main()
