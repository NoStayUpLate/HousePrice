import pandas as pd
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, \
    median_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNetCV,RidgeCV,LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

data_all = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']),ignore_index=True)

# 观察数据情况
# todo train ：过多缺失值，一个个来
# print("train.head():\n",train.head())
# print('train.info():\n',train.info())

# todo test ：过多缺失值，一个个来
# print("test.head():\n",test.head())
# print('test.info():\n',test.info())

# todo data_all:一共2919条数据
# print(data_all.info())

# print(train.describe())

# 观察数据总体情况，利用热力图观察与房间相关性强的因素有哪些
# plt.figure(figsize = (14,12))
# sns.heatmap(train.corr())
# plt.show()

# 获取相关性最高的前N个数据
'''
相关性最高的前10个数据：
'OverallQual' 总体质量
'GrLivArea' 总体居住面积
'GarageCars' 车库里车的数量
'GarageArea' 车库面积
'TotalBsmtSF' 地下室面积
'1stFlrSF' 一层楼的面积
'FullBath' 卫生间数量
'TotRmsAbvGrd'
'YearBuilt' 修建年限
'YearRemodAdd' 重建年份
'''
CORR = {}
correlation = train.corr()
# 置为11，是因为要去除SalePrice本身
N = 11
corVlaues = correlation.nlargest(n=N,columns='SalePrice')['SalePrice'].values
corIndex = correlation.nlargest(n=N,columns='SalePrice')['SalePrice'].index
for i,j in zip(corIndex,corVlaues):
    CORR[i] = j
del CORR['SalePrice']

# 逐一观察相关联的性质：有可能是正相关，也有可能是负相关
# todo OverallQual:总体质量的评定值越高房价越高
# plt.bar(train.OverallQual,train.SalePrice)
# plt.show()

# todo GarageCars:The highest is 3,1-3 is positive,4 is equal to 2
# plt.bar(train.GarageCars,train.SalePrice)
# plt.show()

# 观察GrLivArea是否为正态分布，观察偏度和峰度
# fig = plt.figure()
# sns.distplot(train['GrLivArea'])
# plt.show()
# stats.probplot(train['GrLivArea'],plot = plt)
# plt.show()

# a = pd.Series(data_all.columns)
# BsmtList = a[a.str.contains('Bsmt')].values
# condition = (data_all['BsmtExposure'].isnull()) & (data_all['BsmtCond'].notnull())
# print(data_all[condition][BsmtList])

# 数据清洗
def missing_data(data_all):
    all_data_na = pd.DataFrame(data_all.isnull().sum(),columns={'MissingNum'})
    all_data_na['MissingRatio'] = all_data_na.MissingNum/len(data_all)*100
    all_data_na['ExistNum'] = len(data_all) - all_data_na.MissingNum
    all_data_na['train_notna'] = len(train) - train.isnull().sum()
    all_data_na['test_notna'] = all_data_na['ExistNum'] - all_data_na['train_notna']
    all_data_na['dtype'] = all_data_na.dtypes
    all_data_na = all_data_na[all_data_na.MissingNum > 0].reset_index().sort_values(by=['MissingNum','index'],ascending=False)
    all_data_na.set_index('index', inplace=True)

    return all_data_na

data_all_na = missing_data(data_all)
# print(data_all_na)
# print(data_all.info())
# todo PoolQC的缺失值
data_all.PoolQC = data_all.PoolQC.fillna("No")

# todo Fence FirePlaceQu Alley MiscFeature
data_all.Fence = data_all.Fence.fillna("No")
data_all.FireplaceQu = data_all.FireplaceQu.fillna("No")
data_all.Alley = data_all.Alley.fillna("No")
data_all.MiscFeature = data_all.MiscFeature.fillna("No")

# todo Grage*
data_all[['GarageCond', 'GarageFinish', 'GarageQual', 'GarageType']] = data_all[['GarageCond', 'GarageFinish', 'GarageQual', 'GarageType']].fillna('None')
data_all[['GarageCars', 'GarageArea']] = data_all[['GarageCars', 'GarageArea']].fillna(0)
data_all['Electrical'] = data_all['Electrical'].fillna(data_all['Electrical'].mode()[0])

data_all.GarageYrBlt = data_all.GarageYrBlt.fillna(0)

# todo Bsmt*
a = pd.Series(data_all.columns)
BsmtList = a[a.str.contains('Bsmt')].values

condition = (data_all['BsmtExposure'].isnull()) & (data_all['BsmtCond'].notnull())  # 3个
data_all.ix[(condition), 'BsmtExposure'] = data_all['BsmtExposure'].mode()[0]

condition1 = (data_all['BsmtCond'].isnull()) & (data_all['BsmtExposure'].notnull())  # 3个
data_all.ix[(condition1), 'BsmtCond'] = data_all.ix[(condition1), 'BsmtQual']

condition2 = (data_all['BsmtQual'].isnull()) & (data_all['BsmtExposure'].notnull())  # 2个
data_all.ix[(condition2), 'BsmtQual'] = data_all.ix[(condition2), 'BsmtCond']

# 对于BsmtFinType1和BsmtFinType2
condition3 = (data_all['BsmtFinType1'].notnull()) & (data_all['BsmtFinType2'].isnull())
data_all.ix[condition3, 'BsmtFinType2'] = 'Unf'

allBsmtNa = data_all_na.ix[BsmtList, :]
allBsmtNa_obj = allBsmtNa[allBsmtNa['dtype'] == 'object'].index
allBsmtNa_flo = allBsmtNa[allBsmtNa['dtype'] != 'object'].index
data_all[allBsmtNa_obj] = data_all[allBsmtNa_obj].fillna('None')
data_all[allBsmtNa_flo] = data_all[allBsmtNa_flo].fillna(0)

# todo MasVnr*
data_all.MasVnrType = data_all.MasVnrType.fillna("None")
data_all.MasVnrArea = data_all.MasVnrArea.fillna(0)

# todo MSZoning fillna with mode
data_all.MSZoning = data_all.groupby("MSSubClass")['MSZoning'].transform(lambda x:x.fillna(x.mode()[0]))

# todo LotFrontage fillna with mean
data_all.LotFrontage = data_all.groupby("BldgType")['LotFrontage'].transform(lambda x:x.fillna(x.mean()))

# todo 其他
data_all['KitchenQual'] = data_all['KitchenQual'].fillna(data_all['KitchenQual'].mode()[0])
data_all['Exterior1st'] = data_all['Exterior1st'].fillna(data_all['Exterior1st'].mode()[0])
data_all['Exterior2nd'] = data_all['Exterior2nd'].fillna(data_all['Exterior2nd'].mode()[0])
data_all["Functional"] = data_all["Functional"].fillna(data_all['Functional'].mode()[0])
data_all["SaleType"] = data_all["SaleType"].fillna(data_all['SaleType'].mode()[0])
data_all["Utilities"] = data_all["Utilities"].fillna(data_all['Utilities'].mode()[0])

# na1 = missing_data(data_all)
# print(na1)

# 特征工程
# todo 相关性分析
y_train = train.SalePrice.values
train.drop(['SalePrice'],axis = 1,inplace = True)
corr_matrix = train.corr().abs()
# plt.figure(figsize=(15,10))
# sns.heatmap(corr_matrix,cmap='spectral')
# plt.show()
# 将相关性大于0.75的挑选出来
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
to_drop = [i for i in upper.columns if any(upper[i] >= 0.75)]
# print(to_drop)

data_all.drop(to_drop,axis = 1,inplace = True)
# print(data_all.shape)

# todo 异常值处理
# sns.regplot(x = train.GrLivArea,y = y_train)
# plt.show()
# 观察发现，有面积大但价格却偏低的数据，删除，否则会影响后续模型的训练
dropplot = train.sort_values(by = 'GrLivArea',ascending = False)[:2].GrLivArea
drop_point_list = dropplot.index.tolist()
data_all.drop(drop_point_list,inplace = True)
y_train = np.delete(y_train,drop_point_list)
data_train_index = train.shape[0] - len(drop_point_list)

# todo 特征转换
# print(data_all.groupby('MSSubClass')['MSSubClass'].count())
data_all.MSSubClass = data_all.MSSubClass.apply(str)
data_all.MSSubClass = LabelEncoder().fit_transform(data_all.MSSubClass)
# print(data_all.MSSubClass)

# 观察数值是否异常
skew_tresh = 0.5
skewed = data_all.skew().sort_values(ascending=False)
# 将绝对偏度值大于0.5的数据做平滑处理，用以更好的适应训练模型
skewed_Count = skewed[abs(skewed) > skew_tresh]
skewed_index_list = skewed_Count.index.tolist()
data_all[skewed_index_list] = data_all[skewed_index_list].apply(np.log1p)
# print(data_all[skewed_index_list].head())

# 将文本类型的变量转换成数据类型
# 使用one—hot编码，将数据转换成能够更好适应训练模型的0、1分类
categories = data_all.select_dtypes(include = ['object'])
categories.columns.tolist()
data_all = pd.get_dummies(data_all)
y_train = np.log1p(y_train)
# 将数据控制在一个标准差内，以更好的适应训练模型
data_all = (data_all - data_all.mean())/(data_all.max() - data_all.min())

x_train = data_all.loc[:data_train_index+1]
x_test = data_all.loc[data_train_index+2:]


def _ApplyLinerAlgo(model,x_train,x_test,y_train):
    model.fit(x_train,y_train)
    y_predict = model.predict(x_train)
    print("r2评价模型好坏：",r2_score(y_train,y_predict))
    print("MSE评价模型好坏：",mean_squared_error(y_train,y_predict))
    print("MAE评价模型好坏：",mean_absolute_error(y_train,y_predict))
    print("EAS评价模型好坏：",explained_variance_score(y_train,y_predict))
    print("MAE评价模型好坏：",median_absolute_error(y_train,y_predict))
    print('\n')
    y_train_pre = np.exp(model.predict(x_test))
    return y_train_pre
# 使用弹性回归模型训练和预测
ENCV = ElasticNetCV(alphas=[0.0001,0.0005,0.001,0.01,0.1,1,10],l1_ratio= [0.01,0.1,0.5,0.9,0.99],max_iter=20)
print("ElasticNetCV:\n")
y_pre_ENCV = _ApplyLinerAlgo(ENCV,x_train,x_test,y_train)

# 使用岭回归模型训练和预测
# RCV = RidgeCV(alphas=[0.0001,0.0005,0.001,0.01,0.1,1,10])
# print("RidgeCV:\n")
# y_pre_RCV = _ApplyLinerAlgo(RCV,x_train,x_test,y_train)

# 使用随机森岭回归模型训练和预测
# RFR = RandomForestRegressor()
# print("RandomForestRegressor:\n")
# y_pre_RFR = _ApplyLinerAlgo(RFR,x_train,x_test,y_train)

# 使用Lasso回归模型训练和预测
# LCV = LassoCV(alphas=[0.0001,0.0005,0.001,0.01,0.1,1,10])
# print("LassoCV:\n")
# y_pre_LCV = _ApplyLinerAlgo(LCV,x_train,x_test,y_train)

# 选择两个模型进行组合
pipe = Pipeline([('select',SelectKBest(k='all')),
                 ('ElasticNetCV',ElasticNetCV(alphas = [0.0001,0.0005,0.001,0.01,0.1,1,10],l1_ratio= [0.01,0.1,0.5,0.9,0.99]))])
# 选择要迭代的参数，寻找最优特征值个数
# param_test = {'ElasticNetCV__max_iter':list(range(10,500,10))}
# # 网格搜索寻找最佳参数值
# gridsearch = GridSearchCV(estimator=pipe,param_grid=param_test,scoring='neg_mean_absolute_error',cv=10)
# gridsearch.fit(x_train,y_train)
# print(gridsearch.best_params_,gridsearch.best_score_)
# 贝叶斯搜索
# Bayessearch = BayesSearchCV(estimator=pipe,fit_params=param_test,search_spaces={'C': (0.01, 100.0, 'log-uniform')},scoring='r2',cv=10)
# Bayessearch.fit(x_train,y_train)
# print(Bayessearch.best_params_,Bayessearch.best_score_)
# 将选出的参数带入数据模型中进行训练
selectKBest = SelectKBest(k='all')
encv = ElasticNetCV(alphas=[0.0001,0.0005,0.001,0.01,0.1,1,10],l1_ratio= [0.01,0.1,0.5,0.9,0.99],max_iter=20)
pipeline = make_pipeline(selectKBest,encv)

print("Pipeline:\n")
y_pre_Pipe = _ApplyLinerAlgo(pipeline,x_train,x_test,y_train)

# 利用交叉验证预估模型准确率
cv_score = cross_val_score(pipeline,x_train,y_train,cv = 10)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))
# 预测
prediction = pipeline.predict(x_test)

id = test.Id
submission = pd.DataFrame({'Id':id,'SalePrice':y_pre_Pipe})
submission.to_csv('./HousePriceSubmission.csv',index=False)
print(submission.head(5))