# 题目 1：斐波那契数列题
# 任务： 编写一个程序，接收用户输入的整数 N，生成斐波那契数列的前 N 个数字。
# # 示例输入/输出：
# # 请输入要生成的斐波那契数列个数（N）：10
# # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
num=int(input())
arr=[]
arr.append(0)
arr.append(1)
for i in range(num-2):
    length=len(arr)
    arr.append(arr[length-1]+arr[length-2])
print(arr)


# 题目 2：目标检测数据的解析与过滤
# 背景： 在训练视觉模型时，我们经常需要处理 JSON 格式的标注数据或推理结果。
# 任务： 假设你接收到了一个由后端 API 返回的图像目标检测结果，包含多个预测框的坐标
# y_min, x_max, y_max] 和置信度
# 1. 过滤掉所有置信度
# score 。 编写一个函数，完成以下任务：
# score 小于 0.75 的预测框。
# 2. 计算剩余每个预测框的中心点坐标
# (x_center, y_center) 。
# 3. 返回一个新列表，只包含过滤后框的中心点坐标。
# # 示例输入数据：
# detections = [
# {"class": "crack", "score": 0.85, "bbox": [100, 150, 200, 250]},
# {"class": "rust", "score": 0.60, "bbox": [50, 50, 90, 90]},
# {"class": "crack", "score": 0.92, "bbox": [300, 400, 350, 480]}
# ]
# [x_min,
# # 期望输出：
# # [(150.0, 200.0), (325.0, 440.0)
def filter_detections(detections):
    result = []
    for item in detections:
        score = item["score"]
        if score < 0.75:
            continue
        x_min, y_min, x_max, y_max = item["bbox"]
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        result.append((x_center, y_center))
    return result



# 题目 3：数据集的脏数据清洗
# 背景： 在构建自定义数据集时，收集到的文件后缀往往不统一，直接输入 DataLoader 容易报错。
# 任务： 编写一个函数，接收一个包含多个文件名的列表。要求：
# 1. 剔除所有非图片格式的文件（仅保留包含
# jpg ,
# jpeg ,
# png 后缀的文件，忽略大小写）。
# 2. 将保留下来的图片文件统一重命名为标准格式，如
# dataset_0001.jpg ,
# dataset_0002.jpg 。
# # 示例输入数据：
# raw_files = ["IMAGE1.JPEG", "data.txt", "pic2.png", "script.py",
# "bridge_view.Jpg"]
# # 期望输出：
# # ["dataset_0001.jpg", "dataset_0002.jpg", "dataset_0003.jpg"]
def pictures_only(raw_image):
    suffix_arr=["jpg","jpeg","png"]
    result=[]
    for item in raw_image:
        suffix=item.split(".")[-1].lower()
        if suffix in suffix_arr:
            result.append(item)
    true_result=[]
    for idx in range(len(result)):
        new_name=f"dataset_{idx+1:04d}.jpg"
        true_result.append(new_name)
    return true_result



# 题目 4：文本文档统计题
# 任务： 编写一个程序，读取一个本地的文本文件（
# .txt ），统计并输出该文件中的总行数、总单词数
# （以空白字符分隔为准）以及总字符数。
# 注：请自行创建一个包含若干段落的英文语料测试文件，并在提交作业时与代码一并提交。
# # 示例输出：
# # 语料库统计结果：
# # 行数： 15
# # 单词数： 234
# # 字符数： 1456
with open("input.txt","r") as f:
    content=f.read()
    char_num=len(content)
    word_num=len(content.split(" "))
    line_num = len(content.splitlines())
    print("语料库统计结果")
    print(f"行数： {line_num}")
    print(f"单词数： {word_num}")
    print(f"字符数： {char_num}")





