import csv
from datetime import datetime

from matplotlib import pyplot as plt

# 从文件中获取日期和最高气温
filename = 'models/dsb2018_96_NestedUNet_woDS/log.csv'
with open(filename) as f:
    reader = csv.reader(f)  # 按行读取csv文件中的数据
    header_row = next(reader)  # 返回文件下一行，这里即第一行

    epoch, loss, val_loss = [], [], []
    for row in reader:
            epoch.append(float(row[0]))
            loss.append(float(row[2]))  # 最高气温的索引为1，为方便绘表用int将字符串转化为数字
            val_loss.append(float(row[4]))


# 根据数据绘制图形
fig = plt.figure(dpi=128, figsize=(10, 6))  # dpi分辨率，figsize窗口尺寸，jupyter好像没区别
plt.plot(epoch, loss, c='red', alpha=0.5)  # alpha不透明度，0-1
plt.plot(epoch, val_loss, c='blue', alpha=0.5)
# 设置图形的格式
plt.title("loss funtion", fontsize=20)
plt.xlabel('', fontsize=16)
fig.autofmt_xdate()  # 调用fig.autofmt_xdate()来绘制斜的的日期标签
#plt.ylabel("Temperature (F)", fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)

plt.show()