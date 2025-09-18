#!/usr/bin/env python3
"""
位置编码测试运行脚本
快速运行位置编码的功能测试
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 添加src目录到Python路径
sys.path.append(os.path.join(project_root, 'src'))

def run_basic_test():
    """运行基础测试（不需要可视化依赖）"""
    print("🧪 运行基础功能测试...")
    
    try:
        from tests.functional.test_positional_encoding import (
            test_positional_encoding_basic,
            test_positional_encoding_different_lengths,
            test_positional_encoding_parameters,
            test_positional_encoding_math_properties,
            benchmark_positional_encoding
        )
        
        # 运行基础测试
        test_positional_encoding_basic()
        test_positional_encoding_different_lengths()
        test_positional_encoding_parameters()
        test_positional_encoding_math_properties()
        benchmark_positional_encoding()
        
        print("✅ 基础测试全部通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def run_visualization_test():
    """运行可视化测试"""
    print("🎨 运行可视化测试...")
    
    try:
        from tests.functional.test_positional_encoding import (
            visualize_positional_encoding,
            visualize_positional_encoding_3d
        )
        
        # 检查依赖
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from sklearn.decomposition import PCA
        except ImportError as e:
            print(f"⚠️ 缺少可视化依赖: {e}")
            print("请安装: pip install matplotlib scikit-learn")
            return False
        
        # 运行可视化测试
        visualize_positional_encoding()
        visualize_positional_encoding_3d()
        
        print("✅ 可视化测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 可视化测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 位置编码测试运行器\n")
    
    # 运行基础测试
    basic_success = run_basic_test()
    
    if basic_success:
        print("\n" + "="*50)
        
        # 询问是否运行可视化测试
        try:
            response = input("是否运行可视化测试？(y/n): ").lower().strip()
            if response in ['y', 'yes', '是']:
                run_visualization_test()
            else:
                print("跳过可视化测试")
        except KeyboardInterrupt:
            print("\n测试被用户中断")
        except:
            print("无法获取用户输入，跳过可视化测试")
    
    print("\n🎉 测试完成！")

if __name__ == "__main__":
    main()
