import os
import json

def convert_txt_to_json(input_dir, output_file):
    all_data = []  # 用列表收集所有数据，最后统一转为JSON数组
    global_image_id = 0
    
    val_files = [f for f in os.listdir(input_dir) 
                if f.endswith('_val.txt') and os.path.isfile(os.path.join(input_dir, f))]
    
    for txt_file in val_files:
        file_path = os.path.join(input_dir, txt_file)
        category = txt_file[:-8]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                caption = line.strip()
                if caption:
                    all_data.append({
                        "image_id": global_image_id,
                        "caption": caption,
                        "category": category
                    })
                    global_image_id += 1
    
    # 写入JSON数组（带[]和逗号）
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)  # indent=2美化格式
    
    print(f"转换完成，共{global_image_id}条数据，已保存到{output_file}")

if __name__ == "__main__":
    input_directory = "/home/leihaodong/ICLR25/sjdtree/data/prompts/T2I-CompBench_dataset"
    output_json = "./T2I-CompBench_val.json"  # 输出标准JSON文件
    convert_txt_to_json(input_directory, output_json)