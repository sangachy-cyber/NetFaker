# 损伤仪模块使用示例

"""
本示例展示了如何使用损伤仪模块与HoloWAN设备进行交互。
包括设备连接、路径管理、损伤配置等功能的使用方法。
"""

from netfaker.impairment import ImpairmentFactory


def main():
    """主函数，展示损伤仪模块的完整使用流程"""
    # 损伤仪配置
    HOLowan_ip = "192.168.1.111"
    HOLowan_port = "8080"
    engine_id = 1
    path_id = 1
    
    # 1. 创建损伤仪实例
    print("1. 创建HoloWAN损伤仪实例...")
    impairment_device = ImpairmentFactory.create_device(
        device_type="holowan",
        ip=HOLowan_ip,
        port=HOLowan_port
    )
    
    if not impairment_device:
        print("创建损伤仪实例失败，退出程序")
        return
    
    # 2. 连接设备
    print("\n2. 连接HoloWAN设备...")
    if impairment_device.connect():
        print(f"成功连接到HoloWAN设备: {HOLowan_ip}:{HOLowan_port}")
    else:
        print(f"连接HoloWAN设备失败: {HOLowan_ip}:{HOLowan_port}")
        return
    
    try:
        # 3. 获取设备信息
        print("\n3. 获取设备信息...")
        device_info = impairment_device.get_device_info()
        if device_info["success"]:
            print(f"设备信息获取成功")
        else:
            print(f"设备信息获取失败: {device_info.get('error')}")
        
        # 4. 获取引擎信息
        print(f"\n4. 获取引擎信息 (engine_id={engine_id})...")
        engine_info = impairment_device.get_engine_info(engine_id)
        if engine_info["success"]:
            print(f"引擎信息获取成功")
        else:
            print(f"引擎信息获取失败: {engine_info.get('error')}")
        
        # 5. 创建路径
        print(f"\n5. 创建路径 (engine_id={engine_id}, path_id={path_id})...")
        create_result = impairment_device.create_path(
            engine_id=engine_id,
            path_id=path_id,
            path_name="TestPath"
        )
        if create_result["success"]:
            print(f"路径创建成功")
        else:
            print(f"路径创建失败: {create_result.get('error')}")
        
        # 6. 设置路径属性
        print(f"\n6. 设置路径属性 (engine_id={engine_id}, path_id={path_id})...")
        set_property_result = impairment_device.set_path_property(
            engine_id=engine_id,
            path_id=path_id,
            name="UpdatedTestPath"
        )
        if set_property_result["success"]:
            print(f"路径属性设置成功")
        else:
            print(f"路径属性设置失败: {set_property_result.get('error')}")
        
        # 7. 应用损伤配置
        print(f"\n7. 应用损伤配置 (engine_id={engine_id}, path_id={path_id})...")
        impairment_params = {
            "direction": 1,  # 仅损伤下行
            "delay": {
                "type": "normal",
                "minimum": 10,
                "mean": 100,
                "std_deviation": 20,
                "enable_reordering": 1
            },
            "loss": {
                "type": "burst",
                "probability": 5,
                "min": 1,
                "max": 3
            },
            "bandwidth": {
                "type": "fixed",
                "rate": 100,
                "unit": 1  # Mbps
            }
        }
        apply_result = impairment_device.apply_impairment(
            engine_id=engine_id,
            path_id=path_id,
            impairment_params=impairment_params
        )
        if apply_result["success"]:
            print(f"损伤配置应用成功")
        else:
            print(f"损伤配置应用失败: {apply_result.get('error')}")
        
        # 8. 设置单个损伤参数
        print(f"\n8. 设置单个损伤参数 (engine_id={engine_id}, path_id={path_id})...")
        set_param_result = impairment_device.set_impairment_param(
            engine_id=engine_id,
            path_id=path_id,
            param_name="delay",
            param_value=150
        )
        if set_param_result["success"]:
            print(f"损伤参数设置成功")
        else:
            print(f"损伤参数设置失败: {set_param_result.get('error')}")
        
        # 9. 获取路径统计信息
        print(f"\n9. 获取路径统计信息 (engine_id={engine_id}, path_id={path_id})...")
        stats_result = impairment_device.get_path_stats(
            engine_id=engine_id,
            path_id=path_id
        )
        if stats_result["success"]:
            print(f"路径统计信息获取成功")
        else:
            print(f"路径统计信息获取失败: {stats_result.get('error')}")
        
        # 10. 查询所有路径
        print(f"\n10. 查询所有路径 (engine_id={engine_id})...")
        all_paths_result = impairment_device.get_all_paths(engine_id)
        if all_paths_result["success"]:
            print(f"查询所有路径成功，共 {len(all_paths_result['data'])} 条路径")
            for path_id, path_info in all_paths_result["data"].items():
                print(f"  路径ID: {path_id}, 名称: {path_info['path_name']}, 启用状态: {path_info['is_enable']}")
        else:
            print(f"查询所有路径失败: {all_paths_result.get('error')}")
        
        # 11. 检查路径是否存在
        print(f"\n11. 检查路径是否存在 (engine_id={engine_id}, path_id={path_id})...")
        exists = impairment_device.path_exists(engine_id, path_id)
        print(f"路径 {path_id} {'存在' if exists else '不存在'}")
        
        # 12. 查询所有分类器
        print(f"\n12. 查询所有分类器 (engine_id={engine_id})...")
        all_classifiers_result = impairment_device.get_all_classifiers(engine_id)
        if all_classifiers_result["success"]:
            classifiers_data = all_classifiers_result["data"]
            print(f"查询所有分类器成功")
            print(f"  分类器数据: {classifiers_data}")
        else:
            print(f"查询所有分类器失败: {all_classifiers_result.get('error')}")
        
        # 13. 检查分类器是否存在
        print(f"\n13. 检查分类器是否存在 (engine_id={engine_id}, classifier_id=1)...")
        classifier_exists = impairment_device.classifier_exists(engine_id, 1)
        print(f"分类器 1 {'存在' if classifier_exists else '不存在'}")
        
        # 14. 根据路径名称查询路径ID
        print(f"\n14. 根据路径名称查询路径ID (engine_id={engine_id})...")
        path_name_to_find = "TestPath"
        path_id_by_name = impairment_device.get_path_id_by_name(engine_id, path_name_to_find)
        if path_id_by_name:
            print(f"路径名称 '{path_name_to_find}' 对应的路径ID是: {path_id_by_name}")
        else:
            print(f"未找到名称为 '{path_name_to_find}' 的路径")
        
        # 15. 根据分类器名称查询分类器ID
        print(f"\n15. 根据分类器名称查询分类器ID (engine_id={engine_id})...")
        classifier_name_to_find = "test_classifier"
        classifier_id_by_name = impairment_device.get_classifier_id_by_name(engine_id, classifier_name_to_find)
        if classifier_id_by_name:
            print(f"分类器名称 '{classifier_name_to_find}' 对应的分类器ID是: {classifier_id_by_name}")
        else:
            print(f"未找到名称为 '{classifier_name_to_find}' 的分类器")
        
        # 16. 清除损伤配置
        print(f"\n14. 清除损伤配置 (engine_id={engine_id}, path_id={path_id})...")
        clear_result = impairment_device.clear_impairment(
            engine_id=engine_id,
            path_id=path_id
        )
        if clear_result["success"]:
            print(f"损伤配置清除成功")
        else:
            print(f"损伤配置清除失败: {clear_result.get('error')}")
        
        # 15. 重置引擎
        print(f"\n15. 重置引擎 (engine_id={engine_id})...")
        reset_result = impairment_device.reset_engine(engine_id=engine_id)
        if reset_result["success"]:
            print(f"引擎重置成功")
        else:
            print(f"引擎重置失败: {reset_result.get('error')}")
            
    finally:
        # 12. 断开连接
        print("\n12. 断开设备连接...")
        if impairment_device.disconnect():
            print(f"成功断开与HoloWAN设备的连接")
        else:
            print(f"断开HoloWAN设备连接失败")
    
    print("\n程序执行完成")


if __name__ == "__main__":
    main()
