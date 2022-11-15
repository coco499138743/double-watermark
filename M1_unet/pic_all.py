import csv
from matplotlib import pyplot as plt

filename = 'log/log_unet.csv'
with open(filename) as f:
    reader = csv.reader(f)  # 按行读取csv文件中的数据
    header_row = next(reader)  # 返回文件下一行，这里即第一行

    epoch, val_iou= [], []
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
            val_iou.append(float(row[4]))
filename = 'log/log_nestedunet.csv'
with open(filename) as f:
    reader = csv.reader(f)  # 按行读取csv文件中的数据
    header_row = next(reader)  # 返回文件下一行，这里即第一行
    val_iou1 = []
    for row in reader:
        val_iou1.append(float(row[4]))


filename = 'log/log_fcn.csv'
with open(filename) as f:
    reader = csv.reader(f)  # 按行读取csv文件中的数据
    header_row = next(reader)  # 返回文件下一行，这里即第一行
    val_iou2 = []
    for row in reader:
        val_iou2.append(float(row[4]))
filename = 'log/fcn.csv'
with open(filename) as f:
    reader = csv.reader(f)  # 按行读取csv文件中的数据
    header_row = next(reader)  # 返回文件下一行，这里即第一行
    iou2 = []
    for row in reader:
        iou2.append(float(row[5]))

filename = 'log/unet++.csv'
with open(filename) as f:
    reader = csv.reader(f)  # 按行读取csv文件中的数据
    header_row = next(reader)  # 返回文件下一行，这里即第一行
    iou = []
    for row in reader:
        iou.append(float(row[5]))

filename = 'log/unet.csv'
with open(filename) as f:
    reader = csv.reader(f)  # 按行读取csv文件中的数据
    header_row = next(reader)  # 返回文件下一行，这里即第一行
    iou1 = []
    for row in reader:
        iou1.append(float(row[5]))

# 根据数据绘制图形,linestyle="--", marker="*"
fig = plt.figure(dpi=500, figsize=(10, 8))
plt.plot(epoch, iou, c='r', linestyle='--',marker='o',label="unwatermarked_unet++")
plt.plot(epoch, iou1, c='green',linestyle='--', label="unwatermarked_unet")
plt.plot(epoch, iou2, c='blue',linestyle='--',marker='>', label="unwatermarked_fcn")
plt.plot(epoch, val_iou, c='r',marker='o', label="watermarked_unet++")
plt.plot(epoch, val_iou1, c='green',label="watermarked_unet")
plt.plot(epoch, val_iou2, c='blue', marker='>', label="watermarked_fcn")
plt.legend(loc='lower right', fontsize=12, frameon=True, fancybox=True, framealpha=0.2, borderpad=0.3,
           ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)
# 设置图形的格式

plt.title("Model accuracy", fontsize=20)
plt.ylabel("IOU",fontsize=16)
plt.xlabel('Epoch number', fontsize=16)
fig.autofmt_xdate()  # 调用fig.autofmt_xdate()来绘制斜的的日期标签
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('img1/val_dice.png',dpi=500)
plt.show()