# 第四届魔镜杯大赛

[赛题链接](https://ai.ppdai.com/mirror/goToMirrorDetail?mirrorId=17&tabindex=1)

资金流动性管理迄今仍是金融领域的经典问题。在互联网金融信贷业务中，单个资产标的金额小且复杂多样，对于拥有大量出借资金的金融机构或散户而言，资金管理压力巨大，精准地预测出借资金的流动情况变得尤为重要。本次比赛以互联网金融信贷业务为背景，以《现金流预测》为题，希望选手能够利用提供的数据，精准地预测资产组合在未来一段时间内每日的回款金额。

本赛题涵盖了信贷违约预测、现金流预测等金融领域常见问题，同时又是复杂的时序问题和多目标预测问题。希望参赛者利用聪明才智把互联网金融的数据优势转化为行业解决方案。

## 评估指标

在本次比赛中，参赛队伍需要预测测试集中所有标的在成交日至第1期应还款日内日的回款金额，预测金额数据应为非负数且保留四位小数。本次比赛主要期望选手对未知资产组合每日总回款金额数据预测得越准越好，因此，不会给出标的归属资产组合序号，并选用RMSE作为评估指标，计算公式如下：

![](https://aifile.ppdai.com/db57926066ed44e783dbe7e9a2565144..png)

## 思路

首先将问题转化为一个33分类的问题，提取用户还款日距成交日的日期间隔当作分类标签。没逾期的用户共有32个还款日选择，即32个类别，逾期的用户为第33个类别。

在用户静态、动态以及标的三个维度提取190余个特征，包括统计特征、排名特征以及算术特征等。

利用LightGBM模型对特征重要性排序，选取前50个特征再次训练LightGBM模型与XGBoost模型，对两个模型进行简单的加权平均融合，相比于单模型RMSE下降了近200。

## 结果

第二次以赛代学式的参加比赛，初赛成绩114/504，复赛成绩101/126。

## 文件说明

执行顺序

- eda.pdf：探索性数据分析
- feature_extract.py：提取特征
- data_split.py：划分训练集和验证集
- feature_select.py：特征选择
- fillna.py：缺失值处理
- train.py：训练
- score.py：线下验证

辅助文件

- config.py：配置文件
- functions.py：函数文件