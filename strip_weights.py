import torch

def strip_residual_weights(input_path, output_path):
    # 1. 加载原始的完整权重字典
    print(f"正在加载原始权重: {input_path}")
    state_dict = torch.load(input_path)
    
    # 2. 创建一个新字典，专门用来存放过滤后的权重
    cleaned_state_dict = {}
    
    # 3. 遍历原字典，剥离残差层
    for key, value in state_dict.items():
        # 只要键名里不包含 'residual_layer'，我们就保留它（即保留 NODE 和 RBF 层）
        if 'residual_layer' not in key:
            cleaned_state_dict[key] = value
            
    # 4. 保存纯净版的物理基底权重
    torch.save(cleaned_state_dict, output_path)
    
    print("\n✅ 剥离成功！")
    print(f"剥离前的层数: {len(state_dict)}")
    print(f"剥离后的层数: {len(cleaned_state_dict)}")
    print(f"保留的键名: {list(cleaned_state_dict.keys())}")
    print(f"新权重已保存至: {output_path}")

if __name__ == "__main__":
    # 指定输入和输出的文件名
    INPUT_WEIGHTS = '600_residual_6.pth'
    OUTPUT_WEIGHTS = 'physics_base_only.pth'
    
    strip_residual_weights(INPUT_WEIGHTS, OUTPUT_WEIGHTS)