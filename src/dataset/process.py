import json
import os

# 读取txt文件
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 将YOLO格式写入txt文件
def write_yolo_format(file_name, annotations, image_width, image_height):
    with open(file_name, 'w', encoding='utf-8') as file:
        for annotation in annotations:
            x_center = (annotation['x'] + annotation['w'] / 2) / image_width
            y_center = (annotation['y'] + annotation['h'] / 2) / image_height
            width = annotation['w'] / image_width
            height = annotation['h'] / image_height
            class_id = 0  # 假设所有目标属于同一类别，类别ID为0
            file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# 主函数
def main(txt_file_path, output_dir, image_width, image_height):
    # 读取txt文件中的数据
    data = read_txt_file(txt_file_path)
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历数据，为每张图片生成YOLO格式的txt文件
    for image_name, annotations in data.items():
        if annotations:  # 如果有标注
            yolo_file_name = os.path.join(output_dir, image_name.replace('.jpg', '.txt'))
            write_yolo_format(yolo_file_name, annotations, image_width, image_height)
        else:  # 如果没有标注，创建一个空文件
            yolo_file_name = os.path.join(output_dir, image_name.replace('.jpg', '.txt'))
            with open(yolo_file_name, 'w', encoding='utf-8') as file:
                pass  # 创建空文件

# 参数
txt_file_path = '1.txt'  # 替换为您的txt文件路径
output_dir = 'cc'  # 输出目录
image_width = 640
image_height = 360

# 执行
main(txt_file_path, output_dir, image_width, image_height)