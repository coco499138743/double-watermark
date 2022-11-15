import csv
from matplotlib import pyplot as plt

# 从文件中获取日期和最高气温
filename = 'log/log_unet.csv'
with open(filename) as f:
    reader = csv.reader(f)  # 按行读取csv文件中的数据
    header_row = next(reader)  # 返回文件下一行，这里即第一行

    epoch, iou,val_iou= [], [],[]
    i = 0
    '''for row in reader:
        if i < 200:
            epoch.append(float(row[0]))
            loss.append(float(row[5]))
            i = i + 1
        else:
            break'''
    for row in reader:
            epoch.append(float(row[0]))
            iou.append(float(row[2]))
            val_iou.append(float(row[4]))
filename = 'log/log_nestedunet.csv'
with open(filename) as f:
    reader = csv.reader(f)  # 按行读取csv文件中的数据
    header_row = next(reader)  # 返回文件下一行，这里即第一行
    iou1,val_iou1 = [],[]
    for row in reader:
        iou1.append(float(row[2]))
        val_iou1.append(float(row[4]))


filename = 'log/log_fcn.csv'
with open(filename) as f:
    reader = csv.reader(f)  # 按行读取csv文件中的数据
    header_row = next(reader)  # 返回文件下一行，这里即第一行
    iou2,val_iou2 = [],[]
    for row in reader:
        iou2.append(float(row[2]))
        val_iou2.append(float(row[4]))




# 根据数据绘制图形,linestyle="--", marker="*"
fig = plt.figure(dpi=128, figsize=(10, 6))
plt.plot(epoch, iou, c='red', alpha=0.5,label="unet++")
plt.plot(epoch, iou1, c='green', alpha=0.5,label="unet")
plt.plot(epoch, iou2, c='blue', alpha=0.5,label="fcn")
plt.plot(epoch, val_iou, c='red',linestyle=':', alpha=0.5,label="unet")
plt.plot(epoch, val_iou1, c='green', linestyle=':',alpha=0.5,label="unet")
plt.plot(epoch, val_iou2, c='blue', linestyle=':',alpha=0.5,label="unet")
plt.legend(loc='upper right', bbox_to_anchor=(0.2, 0.95))
# 设置图形的格式
plt.title("Model accuracy", fontsize=20)
plt.xlabel('', fontsize=16)
fig.autofmt_xdate()  # 调用fig.autofmt_xdate()来绘制斜的的日期标签
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('img/val_dice.png')
plt.show()