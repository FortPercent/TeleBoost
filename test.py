import pickle

def split_pickle(input_filename, output_filenames, chunk_size=1):
    with open(input_filename, 'rb') as f:
        # 反序列化对象
        data = pickle.load(f)
        
        # 打印 data 的类型，帮助调试
        print("Loaded data type:", type(data))
    
    # 如果 data 是 set 或 dict，转换成列表
    if isinstance(data, (set, dict)):
        print(data)
        data = list(data)
        print(f"Converted data to list, new length: {len(data)}")
    
    # 确保 data 是可切片的（如列表）
    print(len(data))
    if isinstance(data, list):
        # 分块保存
        for i in range(0, len(data), chunk_size):
            chunk_data = data[i:i+chunk_size]
            with open(output_filenames[i // chunk_size], 'wb') as f_chunk:
                pickle.dump(chunk_data, f_chunk)
    else:
        raise TypeError("The data loaded from pickle is not a sliceable type (like a list).")

# 使用示例
input_filename = '/gemini/space/wuxuaner/Dancegrpo/visual_mem_2025_08_20_13_56_16_7.pickle'  # 输入的大文件
output_filenames = ['part1.pkl', 'part2.pkl', 'part3.pkl']  # 拆分后的文件
split_pickle(input_filename, output_filenames)
