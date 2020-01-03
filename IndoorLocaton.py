import numpy as np
import matplotlib.pyplot as plt
import xlrd  # python  读excel 数据的模块
from sklearn import svm  # sklearn provides supported vector machine learning algorthim
from sklearn import ensemble
from sklearn import tree
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def Get_Average(list):
    sum = 0
    for item in list:
        sum += item
    return sum / len(list)


# 训练集
#open_workbook: Open a spreadsheet file for data extraction
# An instance of the :class:`~xlrd.book.Book` class.(返回的是一个实例)
TrainData = xlrd.open_workbook('Training3.xlsx')  # 打开excel 文件读取数据源：
table = TrainData.sheets()[0] #通过索引顺序获取   返回一个xlrd.sheet.Sheet对象
nrows = table.nrows  # 获取有效行数
ncols = table.ncols  #  获取有效的列数
# 第一列为编号， 第二例， 第三列为坐标， 创建要用到的list
TrainX = [([0] * (ncols - 3)) for p in range(nrows - 1)]
TrainY = [([0] * 1) for p in range(nrows - 1)]
TrainCoor = [([0] * 2) for p in range(nrows - 1)]


# 初始化我们的三个list
for i in range(nrows - 1):
    TrainY[i][0] = table.cell(i + 1, 0).value   # 拿到每一列的数据[i][0]==》编号
    for j in range(ncols - 3):
        TrainX[i][j] = table.cell(i + 1, j + 3).value  # 获取所有rssi
    for k in range(2):
        TrainCoor[i][k] = table.cell(i + 1, k + 1).value  # 拿到坐标的值


# print(TrainY)  # 编号
# print(TrainX)  # 所有的RSSI
# print(TrainCoor)  坐标

X = np.array(TrainX)
Y = np.array(TrainY)

# 测试集
TestData = xlrd.open_workbook('TestTingData3.xlsx')
testTable = TestData.sheets()[0]
testNrows = testTable.nrows
testNcols = testTable.ncols
TestX = [([0] * (testNcols - 3)) for p in range(testNrows - 1)]
TestCoor = [([0] * 2) for p in range(testNrows - 1)]
for i in range(testNrows - 1):
    for j in range(testNcols - 3):
        TestX[i][j] = testTable.cell(i + 1, j + 3).value
    for h in range(2):
        TestCoor[i][h] = testTable.cell(i + 1, h + 1).value
onlineX = np.array(TestX)
# print(TestCoor)
actualCoor = np.array(TestCoor)
'''
Desc: 分类  Scikit-Learn库已经实现了所有基本机器学习的算法
参数说明： kernel(核函数类型 rbf:径像核函数/高斯核)
c:错误项的惩罚系数,C越大，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，但是泛化能力降低，也就是对测试数据的分类准确率降低。
gamma: 核函数系数  1 /（n_features *X.var（））作为gamma的值
'''
classifier = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
# 根据给定的训练数据拟合SVM模型
# ravel： 将多维数组降位一维
# fit 是训练函数： 给出RSSI和标签即可训练（标签是坐标）
classifier.fit(X, Y.ravel())  #

y_pre = classifier.predict(onlineX)
# plt.xlabel('Type')
# plt.ylabel('value')
# plt.scatter(y_pre, y_pre)
# plt.show()
plt.xlabel("横坐标")
plt.ylabel("纵坐标")
plt.title('训练集点和测试集点的分布')
Error = []
for i in range(testNrows - 1):
    curPre = int(y_pre[i])   # 拿到预测结果， 并转为int
    PredictCoor = [TrainCoor[curPre - 1][0], TrainCoor[curPre - 1][1]]  # 第1个类别对应TrainCoor[0]的数据，以此类推
    x = PredictCoor[0]
    y = PredictCoor[1]
    plt.scatter(x, y, s=50, c='r', marker='x')

    print(np.linalg.norm(PredictCoor - actualCoor[i, :]))
    Error.append(np.linalg.norm(PredictCoor - actualCoor[i, :]))  # 第i个维度中所有维度的数据（二维）
    print(i, end=" ")
    print("模型预测：", PredictCoor, end=" ")  # 预测的位置
    print("实际值：", actualCoor[i, :], end=" ")  # 实际位置
    print("直线距离：", np.linalg.norm(PredictCoor - actualCoor[i, :]))  # 求二范数：空间上两个向量矩阵的直线距离
print("平准误差：", Get_Average(Error))

# 画散点图


# plt.figure()
for dot in TrainCoor:
    x = dot[0]
    y = dot[1]
    plt.scatter(x, y, s=50, alpha=0.5)
for tdot in TestCoor:
    x1 = tdot[0]
    y1 = tdot[1]
    plt.scatter(x1, y1, s=50, c='k', marker='>')
plt.show()
