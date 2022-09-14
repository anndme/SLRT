import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker
import math

# #lanczos曲线
# x1 = np.arange(-2, 2, 0.1)
# y1 = 0.25*(x1 ** 2) + x1 + 1
# plt.plot(x1,y1,label="Taylor")
#
# x = np.arange(-2, 2, 0.01)
# y=[]
# for t1 in x:
#     y_1 = math.exp(t1)
#     y.append(y_1)
# plt.plot(x, y, label="exp(x)", linestyle="--")
#
# # x2 = np.arange(-2, 2, 0.01)
# # y2=[]
# # for t2 in x:
# #     y_2 = math.exp(1+t2)
# #     y2.append(y_2)
# # plt.plot(x2, y2, label="exp(x)", linestyle="--")
# # x2 = np.arange(-6, 6, 0.1)
# # y2 = 1 / (1 + math.exp(-x2))
# # plt.plot(x2,y2,label="a=3",linestyle="--")
# #plt.plot(x,y3,label="a=5",linestyle="--")
# # plt.title('sinc')
# # plt.ylim(-1, 2.2)  # y轴的范围
# plt.legend()   #打上标签
# plt.show()

# import mpl_toolkits.axisartist as axisartist
# #创建画布
# fig = plt.figure(figsize=(10, 6))
# #使用axisartist.Subplot方法创建一个绘图区对象ax
# ax = axisartist.Subplot(fig, 111)
# #将绘图区对象添加到画布中
# fig.add_axes(ax)
# #通过set_visible方法设置绘图区所有坐标轴隐藏
# ax.axis[:].set_visible(False)
#
# #ax.new_floating_axis代表添加新的坐标轴
# ax.axis["x"] = ax.new_floating_axis(0,0)
# #给x坐标轴加上箭头
# ax.axis["x"].set_axisline_style("->", size = 1.0)
# #添加y坐标轴，且加上箭头
# ax.axis["y"] = ax.new_floating_axis(1,0)
# ax.axis["y"].set_axisline_style("-|>", size = 1.0)
# #设置x、y轴上刻度显示方向
# ax.axis["x"].set_axis_direction("top")
# ax.axis["y"].set_axis_direction("right")
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_xticks([-4,-3,-2,-1,0,1,2,3,4])
# ax.set_yticks([1,2,3,4])



#三曲线图像

# x1 = np.arange(-4, 4, 0.01)
# y1 = 0.085660*(x1 ** 2) -(0.5*x1) + 0.744204
# plt.plot(x1,y1,label="Least squares approximation", color='red')
#
# x2 = np.arange(-4, 4, 0.01)
# y2 = math.log(2)- (0.5*x2) + 0.125*(x2 ** 2)
# plt.plot(x2,y2,label="Taylor approximation", color='green')
#
# x = np.arange(-4, 4, 0.01)
# y=[]
# for t1 in x:
#     y_1 = math.log(1+math.exp(-t1))
#     y.append(y_1)
# plt.plot(x, y, label="log(1+e^z)", linestyle="--")


# x2 = np.arange(-2, 2, 0.01)
# y2=[]
# for t2 in x:
#     y_2 = math.exp(1+t2)
#     y2.append(y_2)
# plt.plot(x2, y2, label="exp(x)", linestyle="--")
# x2 = np.arange(-6, 6, 0.1)
# y2 = 1 / (1 + math.exp(-x2))
# plt.plot(x2,y2,label="a=3",linestyle="--")
#plt.plot(x,y3,label="a=5",linestyle="--")
# plt.title('sinc')
# plt.ylim(-1, 2.2)  # y轴的范围
# plt.legend()   #打上标签
# plt.show()


# x1 = x[0]
# x2 = x[1]
# result = fsolve(f, [1, 1])
# print(result[0], result[1])
# x = np.linspace(0, 20, 1000)  # 作图的自变量x
# y = np.sqrt(2 * x - 1)  # 因变量y
# z = np.square(x) - 2  # 因变量z
# plt.figure(figsize=(8, 4))  # 设置图像大小
# plt.plot(x, y, label='$\sinx+1$', color='red', linewidth=2)  # 作图，设置标签、线条颜色、线条大小
# plt.plot(x, z, 'b--', label='$\cosx^3+1$', linewidth=1)  # 作图，设置标签、
# plt.xlabel('Times(s)')  # x轴名称
# plt.ylabel('Volt')  # y轴名称
# plt.title('求解非线性方程组')  # 标题
# plt.ylim(0, 2.2)  # y轴的范围
# plt.legend()  # 显示图例
# plt.show()



# shops = ["A", "B", "C", "D", "E", "F"]
# sales_product_1 = [100, 85, 56, 42, 72, 15]
# sales_product_2 = [50, 120, 65, 85, 25, 55]
# sales_product_3 = [20, 35, 45, 27, 55, 65]
#
# fig, ax = plt.subplots(figsize=(10, 7))
# # 先创建一根柱子，显示第一种产品的销量
# ax.bar(shops, sales_product_1, color="red", label="Product_1")
# # 第二根柱子“堆积”在第一根柱子上方，通过'bottom'调整，显示第二种产品的销量
# ax.bar(shops, sales_product_2, color="blue", bottom=sales_product_1, label="Product_2")
# #第三根柱子“堆积”在第二根柱子上方，通过'bottom'调整，显示第三种产品的销量
# ax.bar(shops, sales_product_3, color="green",
#        bottom=np.array(sales_product_2) + np.array(sales_product_1), label="Product_3")
#
# ax.set_title("Stacked Bar plot", fontsize=15)
# ax.set_xlabel("Shops")
# ax.set_ylabel("Product Sales")
# ax.legend()
# plt.show()

# dataset = pd.read_csv(r"F:\成绩统计.csv", sep='\s+', header=None) #ACAD
# X = dataset.iloc[:, :-1].values
# Y = dataset.iloc[:, -1].values
# Y = Y.tolist()
#
# print(B)

# import matplotlib.pyplot as plt
# a=[24,23]
# b=[12,23]z
# plt.scatter(a, b)
# plt.show()


# datasets = ["DD", "WIBC", "HDD", "ACAD"] #柱状图
# # SLRS = [1.4037320613861084, 1.6082763671875, 2.144702911376953, 2.4354746341705322]
# VANE = [110.06767988204956,  144.49310779571533, 268.3295121192932, 300.1187963485718]
#
# # 创建分组柱状图，需要自己控制x轴坐标
# xticks = np.arange(len(datasets))
# fig, ax = plt.subplots(figsize=(10, 7))
#
# index=np.arange(len(VANE));
# # ax.bar(xticks, SLRS, width=0.25, label="SLRS", color="indianred")
#
# ax.bar(xticks+0.25, VANE, width=0.25, label="VANE", color="cornflowerblue")
#
#
# # for a,b in zip(index,SLRS):   #柱子上的数字显示
# #  plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=18);
# for a,b in zip(index+0.25,VANE):
#  plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=18);
#
# ax.legend(fontsize=28)
# # 最后调整x轴标签的位置
# ax.set_xticks(xticks+0.25)
# ax.set_xticklabels(datasets, fontsize=18)
# plt.yticks([50, 100, 150, 200, 250, 300], fontsize=18)
# # ax = plt.gca()
# # y 轴用科学记数法
# ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='y', useMathText=True)
#
# plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
# plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
# plt.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=23)
# # ax.set_xticks(xticks + 1)
# # ax.set_xticklabels(datasets)
# plt.grid(axis="y", c = 'grey', linestyle='--')
# plt.show()

# datasets = ["第一题", "第二题", "第三题", "第四题", "第五题"] #柱状图
# 总分 = [20, 30, 20, 10, 20]
# 平均分 = [12.61052632,25.58947368,13.8,3.147368421,15.84210526]
# 最大值 = [18,30,20,8,20]
# 最小值 = [6,18, 0, 0,0]
# 方差 = [4.637783934,10.22094183,16.68631579,3.662493075,21.92243767]
# # 创建分组柱状图，需要自己控制x轴坐标
# xticks = np.arange(len(datasets))
# fig, ax = plt.subplots(figsize=(10, 7))
# # 所有门店第一种产品的销量，注意控制柱子的宽度，这里选择0.25
# index=np.arange(len(总分));
# ax.bar(xticks, 总分, width=0.15, label="总分", color="red")
# # 所有门店第二种产品的销量，通过微调x轴坐标来调整新增柱子的位置
# ax.bar(xticks+0.15, 平均分, width=0.15, label="平均分", color="blue")
# ax.bar(xticks+0.3, 最大值, width=0.15, label="最大值", color="green")
# ax.bar(xticks+0.45, 最小值, width=0.15, label="最小值", color="yellow")
# ax.bar(xticks+0.6, 方差, width=0.15, label="方差", color="purple")
# # 所有门店第三种产品的销量，继续微调x轴坐标调整新增柱子的位置
# # ax.bar(xticks + 0.5, sales_product_3, width=0.25, label="Product_3", color="green")
# for a,b in zip(index,总分):   #柱子上的数字显示
#  plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=7);
# for a,b in zip(index+0.15,平均分):
#  plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=7);
# for a,b in zip(index+0.3,最大值):
#  plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=7);
# for a,b in zip(index+0.45,最小值):
#  plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=7);
# for a,b in zip(index+0.6,方差):
#  plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=7);
#
# # ax.set_title("题目得分情况", fontsize=15)
# # ax.set_xlabel("Datasets")
# ax.set_ylabel("分数")
# ax.legend()
# # 最后调整x轴标签的位置
# ax.set_xticks(xticks+0.3)
# ax.set_xticklabels(datasets)
# #解决中文显示问题
# plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
# plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
# # ax.set_xticks(xticks + 1)
# # ax.set_xticklabels(datasets)
# plt.show()



#随迭代次数变化
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
x = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# y1 = [1.3595187664031982,2.156479835510254,3.0003182888031006,3.8597848415374756,5.219304084777832,5.516211748123169,6.250662565231323,7.375783443450928,6.985145330429077,7.641438722610474] #DD
y2 = [30.460024118423462, 46.56848335266113, 62.29677081108093, 80.09999322891235, 98.88397336006165, 116.12372159957886, 135.2155511379242, 152.93165969848633, 169.76110744476318,
188.42705917358398]
# y1 = [1.9376773834228516,3.2347187995910645,4.984903812408447,5.875625133514404,7.297648906707764,9.01659107208252,10.541746997833252,10.73231430053711,11.032448768615723,12.657594680786133]#ACAD
y3 = [80.79684662818909, 126.5715696811676, 166.82359528541565, 212.2183084487915, 257.92190408706665, 303.5682168006897, 349.77712869644165, 395.0755431652069, 440.96760296821594,
486.5678017139435]
# y1 = [1.5783205032348633,2.687758445739746,3.844189167022705,4.984904050827026,6.110023498535156,7.203890085220337,8.407142162322998,9.500977039337158,10.626152992248535,12.063756465911865,]#HDD
y4 = [69.3714644908905, 107.0155930519104, 147.36783647537231, 187.80639481544495, 228.80341029167175, 274.1998929977417, 315.7834939956665, 358.23457980155945,
407.31221294403076, 450.31221294403076]
# y1 = [1.3907742500305176,2.703411817550659,3.109703779220581,4.109810829162598,5.063037157058716,6.531976699829102,6.969491004943848,7.969563722610474,8.719674825668335,9.78232192993164,]#WIBC
y5 = [37.493547201156616, 57.977184534072876, 79.47130393981934, 99.70816445350647, 117.83430075645447, 136.85892057418823, 159.59838771820068, 183.1726257801056, 205.96671271324158,
225.33216643333435]
plt.plot(x, y2, marker='o', ms=5, label="DD", color='brown',linestyle='--')
plt.plot(x, y3, marker='*', ms=5, label="ACAD", color='cornflowerblue')
plt.plot(x, y4, marker='^', ms=5, label="HDD", color='darkseagreen', linestyle='--')
plt.plot(x, y5, marker='x', ms=5, label="WIBC", color='darkmagenta')

ax = plt.gca()
# y 轴用科学记数法
ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='y', useMathText=True)
ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='x', useMathText=True)

# plt.plot(x, y3, marker='*', ms=10, label="c")
# plt.plot(x, y4, marker='*', ms=10, label="d")
plt.xticks(rotation=0, fontsize=15)
plt.yticks([50, 100, 150, 200, 250, 300], fontsize=15)
plt.xlabel("Number of clients", fontsize=10)
plt.ylabel("Run time(s)", fontsize=12)
# plt.title("Variation of time with number of iterations")
plt.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=8)
plt.grid(axis="y", c = 'grey', linestyle='--')
plt.grid(axis="x", c = 'grey', linestyle='--')
# 在折线图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
# # for y in [y1, y2, y3, y4]:
# for y in [y1, y2]:
#     for x1, yy in zip(x, y):
#         plt.text(x1, yy + 1, str(yy), fontsize=20, rotation=0)
# plt.savefig("a.jpg")
plt.show()

# #随着用户数量
# import matplotlib
# import matplotlib.pyplot as plt
# # matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
# x = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# y1 = [1.217233657836914,1.199160099029541,1.2222509384155273,1.2433135509490967,1.1881790161132812,1.2091872692108154,1.2012252807617188,1.2101871967315674,1.2202446460723877,
# 1.229269027709961] #DD
# y2 = [30.460024118423462, 46.56848335266113, 62.29677081108093, 80.09999322891235, 98.88397336006165, 116.12372159957886, 135.2155511379242, 152.93165969848633, 169.76110744476318,
# 188.42705917358398]
# # y1 = [1.8751986026763916,1.8908586502075195,1.812659740447998,1.8439466953277588,1.9074823856353761, 1.936147689819336,2.2509853839874268,1.969235897064209,1.842900276184082,
# # 1.909076452255249]#ACAD
# # y2 = [80.79684662818909, 126.5715696811676, 166.82359528541565, 212.2183084487915, 257.92190408706665, 303.5682168006897, 349.77712869644165, 395.0755431652069, 440.96760296821594,
# # 486.5678017139435]
# # y1 = [1.8408939838409424,1.8368847370147705,1.8459410667419434,1.8358821868896484,2.3020896911621094,1.8108139038085938,1.7536633014678955,1.8759880065917969,1.8348798751831055,
# # 1.8840415477752686]#HDD
# # y2 = [69.3714644908905, 107.0155930519104, 147.36783647537231, 187.80639481544495, 228.80341029167175, 274.1998929977417, 315.7834939956665, 358.23457980155945,
# # 407.31221294403076, 450.31221294403076]
# # y1 = [1.3595190048217773,1.3751766681671143,1.4064009189605713,1.406367301940918,1.4063987731933594,1.390773057937622,1.4063992500305176,1.390772819519043,1.375180959701538,
# # 1.4063646793365479]#WIBC
# # y2 = [37.493547201156616, 57.977184534072876, 79.47130393981934, 99.70816445350647, 117.83430075645447, 136.85892057418823, 159.59838771820068, 183.1726257801056, 205.96671271324158,
# # 225.33216643333435]
#
# plt.plot(x, y1, marker='o', ms=5, label="SLRS", color='red')
# plt.plot(x, y2, marker='*', ms=5, label="VANE", color='blue')
#
# ax = plt.gca()
# # y 轴用科学记数法
# ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='y', useMathText=True)
# ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='x', useMathText=True)
#
# # plt.plot(x, y3, marker='*', ms=10, label="c")
# # plt.plot(x, y4, marker='*', ms=10, label="d")
# plt.xticks(rotation=0,  fontsize=15)
# plt.yticks([50, 100, 150, 200, 250, 300],  fontsize=15)
# plt.xlabel("Number of clients", fontsize=15)
# plt.ylabel("Run time(s)", fontsize=15)
# # plt.title("Variation of time with number of iterations")
# plt.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=15)
# plt.grid(axis="y", c = 'grey', linestyle='--')
# plt.grid(axis="x", c = 'grey', linestyle='--')
# # 在折线图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
# # # for y in [y1, y2, y3, y4]:
# # for y in [y1, y2]:
# #     for x1, yy in zip(x, y):
# #         plt.text(x1, yy + 1, str(yy), fontsize=20, rotation=0)
# plt.savefig("a.jpg")
# plt.show()

# # 随着样本数量变化
# import matplotlib
# import matplotlib.pyplot as plt
# # matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
# x = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
# # y1 = [0.930474042892456,1.011690616607666,1.089897632598877,1.1069443225860596,1.1721172332763672,1.2142283916473389,1.2613539695739746,2.2249162197113037,1.9301319122314453,
# # 1.6794660091400146,] #DD
# # y2 = [87.75487924575806, 88.44978795051575, 89.65749773979187, 90.02528619766235, 91.60422229766846, 91.88038873672485, 92.13426733016968, 92.83096742630005, 93.11001491546631, 93.38168697357178]
# # y1 = [1.6095459461212158,1.547041416168213,1.593916654586792,1.656426191329956,1.7033061981201172,1.7658116817474365,1.7970664501190186,1.8284239768981934,2.1284804821014404,2.3095998287200928,]#ACAD
# # y2 = [264.9405605792999, 265.42295932769775, 266.60277366638184, 267.52992606163025, 267.64235067367554, 268.0412390232086, 268.5359356403351, 268.55363750457764, 268.788143157959,
# # 268.84638357162476]
# # y1 = [1.5157854557037354,1.4376521110534668,1.4845335483551025,2.1095986366271973,2.9534385204315186,2.203359842300415,1.7814385890960693,1.7658131122589111,1.765812635421753,
# # 1.8908250331878662]#HDD
# # y2 = [222.98452401161194, 223.15705823898315, 223.48583006858826, 223.53073644638062, 223.69413571357727, 223.7617551803589, 223.80868186950684, 223.9549515247345, 223.9976215839386,224.0743151664734]
# # y1 = [0.9844801425933838,1.046985149383545,1.125119924545288,1.1407470703125,1.2032523155212402,1.5782949924468994,2.422130823135376,1.4689042568206787,1.3595199584960938,1.4689066410064697]#WIBC
# # y2 = [112.10565257072449, 112.30893898010254, 112.62165141105652, 112.72570028305054, 112.92143921852112, 113.08086047172546, 113.14303741455078, 113.15931101799011, 114.35321617126465,
# # 114.32853388786316]
#
# plt.plot(x, y1, marker='o', ms=5, label="SLRS", color='red')
# plt.plot(x, y2, marker='*', ms=5, label="VANE", color='blue')
#
# ax = plt.gca()
# # y 轴用科学记数法
# ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='y', useMathText=True)
# ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='x', useMathText=True)
#
# # plt.plot(x, y3, marker='*', ms=10, label="c")
# # plt.plot(x, y4, marker='*', ms=10, label="d")
# plt.xticks(rotation=0, fontsize=15)
# plt.yticks([50, 100, 150, 200, 250, 300,350,400], fontsize=15)
# plt.xlabel("Number of samples",  fontsize=14)
# plt.ylabel("Run time(s)",  fontsize=15)
# # plt.title("Variation of time with number of iterations")
# plt.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=15)
# plt.grid(axis="y", c = 'grey', linestyle='--')
# plt.grid(axis="x", c = 'grey', linestyle='--')
# # 在折线图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
# # # for y in [y1, y2, y3, y4]:
# # for y in [y1, y2]:
# #     for x1, yy in zip(x, y):
# #         plt.text(x1, yy + 1, str(yy), fontsize=20, rotation=0)
# plt.savefig("a.jpg")
# plt.show()


# #预计算时间随样本数量变化
# import matplotlib
# import matplotlib.pyplot as plt
# # matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
# x = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
# y1 = [0.06250548362731934,0.09375715255737305,0.1406402587890625,0.18751740455627441,0.25002479553222656,0.2812800407409668, 0.3906667232513428, 0.8594679832458496,
# 0.59381103515625,0.5625593662261963] #DD
# y2 = [87.74487924575806, 88.34978795051575, 89.55749773979187, 90.02528619766235, 91.60422229766846, 91.88038873672485, 92.13426733016968, 92.83096742630005, 93.11001491546631, 93.33168697357178]
# # y1 = [0.06250619888305664,0.10938763618469238,0.1406404972076416,0.23439931869506836,0.2656524181365967,0.32816028594970703,0.34377360343933105,0.781334638595581,0.453174352645874,
# # 0.7344465255737305]#ACAD
# # y2 = [264.9405605792999, 265.42295932769775, 266.60277366638184, 267.52992606163025, 267.64235067367554, 268.0412390232086, 268.5359356403351, 268.55363750457764, 268.788143157959,
# # 268.84638357162476]
# # y1 = [0.04688096046447754,0.09375905990600586,0.15626788139343262,0.23439860343933105,0.31252312660217285,0.35941123962402344,0.35941314697265625,0.4219193458557129,0.5313053131103516,0.8282122611999512,
# # ]#HDD
# # y2 = [222.98452401161194, 223.15705823898315, 223.48583006858826, 223.53073644638062, 223.69413571357727, 223.7617551803589, 223.80868186950684, 223.9549515247345, 223.9976215839386,
# # 224.0743151664734]
# # y1 = [0.04687952995300293,0.09376096725463867,0.18752121925354004,0.20314764976501465,0.2656538486480713,0.3125338554382324,0.35941433906555176,0.39066529273986816,0.4375464916229248,
# # 1.6251742839813232]#WIBC
# # y2 = [112.10565257072449, 112.30893898010254, 112.62165141105652, 112.72570028305054, 112.92143921852112, 113.08086047172546, 113.14303741455078, 113.15931101799011, 114.35321617126465,
# # 114.32853388786316]
#
# plt.plot(x, y1, marker='o', ms=5, label="SLRS", color='red')
#
# plt.plot(x, y2, marker='*', ms=5, label="VANE", color='blue')
#
# ax = plt.gca()
# # y 轴用科学记数法
# ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='y', useMathText=True)
# ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='x', useMathText=True)
#
# # plt.plot(x, y3, marker='*', ms=10, label="c")
# # plt.plot(x, y4, marker='*', ms=10, label="d")
# plt.xticks(rotation=0,  fontsize=15)
# plt.yticks([50, 100, 150, 200, 250, 300, 350], fontsize=15)
# plt.xlabel("Number of samples",  fontsize=15)
# plt.ylabel("Run time(s)",  fontsize=15)
# # plt.title("Variation of time with number of iterations")
# plt.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=15)
# plt.grid(axis="y", c = 'grey', linestyle='--')
# plt.grid(axis="x", c = 'grey', linestyle='--')
# # 在折线图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
# # # for y in [y1, y2, y3, y4]:
# # for y in [y1, y2]:
# #     for x1, yy in zip(x, y):
# #         plt.text(x1, yy + 1, str(yy), fontsize=20, rotation=0)
# plt.savefig("a.jpg")
# plt.show()




# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import random
# import time
# from phe import paillier
#
# public_key, private_key = paillier.generate_paillier_keypair()
#
#
# class LogisticSecretShare(object):
#
#     def __init__(self, rate, lanta, Y):
#         self.rate = rate
#         self.size = len(Y)
#         self.lanta = lanta
#
#     def generating_matrix(self, x, y, n):
#         A = np.insert(x, 0, [1], axis=0).reshape(n, 1)
#         B = np.append(x, y).reshape(1, n)
#         return np.dot(A, B)  # 矩阵相乘
#
#
#
#     def paillier_(self, number_lists):
#         secret_number_list = list(np.zeros((1, number_lists[0].shape[0] * number_lists[0].shape[1]))[0])
#         encrypted = np.array([public_key.encrypt(x) for x in secret_number_list]).reshape(1, len(secret_number_list))
#
#         for number_list in number_lists:
#             secret_number_list = list(number_list.reshape((1, number_list.shape[1] * number_list.shape[0]))[0])
#             # 加密
#             encrypted += np.array([public_key.encrypt(x) for x in secret_number_list]).reshape(1,
#                                                                                                len(secret_number_list))
#
#         # 解密
#         decrypted = [private_key.decrypt(x) for x in list(encrypted)[0]]
#         decrypted = np.array(decrypted).reshape(number_lists[0].shape[0], number_lists[0].shape[1])
#
#         return decrypted
#
#     start = time.time()
#
#     def split_data(self, X, Y, N):
#
#         self.n = X.shape[1] + 1  # 确定维数
#         chioce = [i for i in range(len(Y))]
#         size = int(len(Y) / N)
#         split_lists = []
#         for i in range(N-1):
#             data = []
#             for j in range(size):
#                 data += [chioce.pop(random.randint(0, len(chioce)-1))]
#             split_lists.append(data)
#         split_lists.append(chioce)
#         paillier_list = []
#
#         for split_list in split_lists:
#             Aik = np.zeros((self.n, self.n))
#             for k in split_list:
#                 Aik += self.generating_matrix(X[k], Y[k], self.n)
#             paillier_list.append(Aik)
#         A = self.paillier_(paillier_list)
#         return A
#
#     def omega_zero(self, omega_init_r, z, u0):
#         omega_new = (1 - 2 * self.lanta * self.rate) * omega_init_r - self.rate / self.size * (0.25 * z - 0.5 * u0)
#         return omega_new
#
#     def generate_V(self, Ar):
#         V0 = Ar[0][:-1]
#         V = [np.insert(V0, 0, self.size)]
#
#         Uj = [float(Ar[0][-1])]
#         for i in range(1, Ar.shape[0]):
#             vj = Ar[i][:-1]
#             V += [np.insert(vj, 0, Ar[0][i - 1])]
#             Uj += [float(Ar[i][-1])]
#         return V, Uj
#
#     def item_model_while(self, itemmax, omega, A):
#         V, Uj = self.generate_V(A)
#         omega_old = omega
#         for u in range(itemmax):
#             omega_new = []
#             for i in range(len(Uj)):
#                 z = np.dot(omega_old,  V[i].T)
#                 omega_new.append(self.omega_zero(omega_old[0][i], z, Uj[i]))
#             omega_old = np.array(omega_new).T
#         return omega_old
#
#
# def load_csv_wisconsin(path):
#     X = []
#     Y = []
#     with open(path, 'r') as f:
#         datas = f.readlines()
#
#     for data in datas:
#         split_data = data.split(',')
#         d = []
#         for i in split_data[1:-1]:
#             try:
#                 d.append(float(i))
#             except:
#                 d.append(np.NAN)
#         X.append(d)
#         if "2" in split_data[-1]:
#             Y.append(1)
#         elif "4" in split_data[-1]:
#             Y.append(-1)
#         else:
#             Y.append(split_data[-1])
#     return np.array(X), np.array(Y)
#
#
# def load_csv_diabetes(path):
#     X = []
#     Y = []
#     with open(path, 'r') as f:
#         datas = f.readlines()
#
#     for data in datas[1:]:
#         split_data = data.split(',')
#         X.append([float(i) for i in split_data[:-1]])
#         if "positive" in split_data[-1]:  # 阳性为1，阴性为-1
#             Y.append(1)
#         elif "negative" in split_data[-1]:
#             Y.append(-1)
#         else:
#             Y.append(split_data[-1])
#     return np.array(X), np.array(Y)
#
#
# def normalization(data):  # 归一化
#     max_ = data.max(axis=0)
#     min_ = data.min(axis=0)
#     diff = max_ - min_
#     zeros = np.zeros(data.shape)
#     m = data.shape[0]
#     zeros = data - np.tile(min_, (m, 1))
#     zeros = zeros / np.tile(diff, (m, 1))
#     return zeros
#
#
# def confusion_matrix(D, X, Y):
#     def sigmoid(inX):
#         return 1.0/(1+np.exp(-inX))
#     omega0 = D[0]
#     omega = D[1:]
#     predict = sigmoid(np.dot(omega, X.T) + omega0) > 0.5
#     Y_predict = Y > 0
#     result = [[np.sum(predict), len(Y) - np.sum(predict)], [np.sum(Y_predict), len(Y)-np.sum(Y_predict)]]
#     return result
#
#
# def accuracy(matrix):
#     return (sum(matrix[1])-abs(matrix[0][0] - matrix[1][0])) / sum(matrix[1])
#
#
#
# if __name__ == '__main__':
#     start = time.time()
#     time_list = []
#     for j in [3, 10, 20]:
#         timeit = []
#         for i in range(1):
#             start = time.time()
#             path = r"C:\Users\张会洋\Desktop\我的论文\breast-cancer-wisconsin.csv"
#             X, Y = load_csv_wisconsin(path)
#             # path = r"C:\Users\张会洋\Desktop\我的论文\diabetes_csv.csv"
#             # X, Y = load_csv_diabetes(path)
#
#             X = normalization(X)
#
#             # from sklearn.cross_validation import train_test_split
#             X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#
#             B = LogisticSecretShare(0.01, 0.0001, Y_train)
#
#             A = B.split_data(X_train, Y_train, j)
#             end = time.time()
#             timeit.append(end - start)
#         print(j, sum(timeit) / len(timeit))
#         time_list.append(sum(timeit) / len(timeit))
#     print(time_list)
#     print('%s,' % (end - start))
#             # for m in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
#     omega0 = np.random.randint(low=np.min(1), high=np.max(10), size=(1, X.shape[1] + 1))
#     omega1 = random.randint(0, 10) - omega0
#
#     D = B.item_model_while(m, omega0, A)
#     matrix = confusion_matrix(D[0], X_train, Y_train)
#     matrix1 = confusion_matrix(D[0], X_test, Y_test)
#     print(accuracy(matrix), matrix)
#     print(accuracy(matrix1), matrix1)
#
#
# # ///////////////////////////////////////////////////
# #     start = time.time()
# #     # path = r"C:\Users\张会洋\Desktop\我的论文\breast-cancer-wisconsin.csv"
# #     # X, Y = load_csv_wisconsin(path)
# #
# #     time_list = []
# #     for j in [3, 50, 60, 70, 120]:
# #         timeit = []
# #         for i in range(100):
# #             start = time.time()
# #             path = r"C:\Users\张会洋\Desktop\我的论文\diabetes_csv.csv"
# #             X, Y = load_csv_diabetes(path)
# #             X = normalization(X)
# #
# #             # from sklearn.crossvalidation import train_test_split
# #             X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# #
# #             A = LogisticSecretShare(0.01, 0.0001, Y_train)
# #             A0, A1 = A.split_data(X_train, Y_train, j)
# #             end = time.time()
# #             timeit.append(end - start)
# #         print(j, sum(timeit) / len(timeit))
# #         time_list.append(sum(timeit) / len(timeit))
# #     print(time_list)
# #     print('Running time: %s Seconds' % (end - start))
# #     omega0 = np.random.randint(low=np.min(1), high=np.max(10), size=(1, X.shape[1] + 1))
# #     omega1 = random.randint(0, 10) - omega0
# #
# #     D = A.item_model_while(1000, omega0, omega1, A0, A1)
# #     matrix = confusion_matrix(D, X_train, Y_train)
# #     matrix1 = confusion_matrix(D, X_test, Y_test)
# #     print(accuracy(matrix), matrix)
# #     print(accuracy(matrix1), matrix1)
#
