# %% [markdown]
# 代码由[kangjiali]编写，对应数据集由北京大学第三医院提供（涉及隐私问题，因此不公开数据集）
# 如有疑问可以联系邮箱（e-mail）：kangjiali@bjmu.edu.cn
# 任务目标：训练机器学习模型,预测并解释pros得分是否达到MCID
# 主要步骤：数据探索和理解-变量筛选-模型构建和评估-模型比较筛选-模型解释

# %% [markdown]
# ## 第一部分：数据理解和特征衍生

# %%
#导入全部所需要的库
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import datetime
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
plt.rcParams [ 'font.sans-serif'] = [ 'SimHei']      #解决中文显示问题
from datetime import datetime # 导入datetime模块

#导入全部所需要的库
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import datetime
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report




# %%
#载入数据
origin_df = pd.read_excel("./model_data0218.xlsx")
origin_df

# %%
# 缺失比例计算
def collect_na_value(dataframe):
    misspor = sum(dataframe.isna().sum()) / (dataframe.shape[0] * dataframe.shape[1])
    print(misspor)


collect_na_value(origin_df)  # 计算出结果
# 数据缺失比例7%，因此可以进行填补（如果超过50%一般很难填补，采取删除方法）

# %%

# 基于KNN算法的缺失值填补
from fancyimpute import KNN  # KNN填充

imputed_data=pd.DataFrame(KNN(k=20, verbose=False).fit_transform(origin_df))
imputed_data.columns = origin_df.columns

collect_na_value(imputed_data) 

# %%
imputed_data

# %% [markdown]
# ## 1.回归的尝试

# %%
y_data = imputed_data[['mHHS改变值','ADL改变值','iHOT12改变值','VAS改变值']]
y_data
   
y_col = ['mHHS改变值','ADL改变值','iHOT12改变值','VAS改变值']   # 必须确保在预测某个Y的时候，其余的Y不要出现（因为他们高度相关）
MCID_col = ['mHHS_MCID','ADL_MCID',	'iHOT12_MCID','VAS_MCID']   # 这些属于事后信息，不能出现在预测的里面，否则就是作弊。
X = imputed_data.copy()
X.drop(columns=y_col,inplace=True)
X.drop(columns=MCID_col,inplace=True)


# %%
print(X)

# %% [markdown]
# ## 2.3 模型训练与评估

# %%
import warnings
import sklearn
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.utils import resample

# 定义LASSO模型和参数网格
def fit_lasso_model(X,Y,y_name):
    lasso = Lasso(max_iter=10000)
    param_grid = {'alpha': [0.1, 1.0, 5,10.0,15,20,30,50]}  
    #alpha 参数控制正则化的强度。较大的 alpha 值意味着更强的正则化，会使得更多的系数变成零，

    # 使用网格搜索寻找最佳参数
    grid_search = GridSearchCV(lasso, param_grid, cv=10)
    grid_search.fit(X, Y)

    # 获取最佳参数和最佳模型
    best_alpha = grid_search.best_params_['alpha']
    best_model = grid_search.best_estimator_

    # 打印最佳参数
    print(f"Best alpha: {best_alpha}")

    # 保存最佳模型
    filename = f'./{y_name}_best_lasso_model_{best_alpha}.pkl'
    joblib.dump(best_model, filename)
    print(f"Best model saved as {filename}")
    model_out = f'./{y_name}_best_lasso_model_{best_alpha}.pkl'
    return  model_out


## 实现点估计+置信区间
# 定义引导函数
def bootstrap_prediction(model, X, n_iterations=100):
    # 存储每次引导的预测结果
    bootstrap_preds = np.zeros((X.shape[0], n_iterations))

    for i in range(n_iterations):
        # 对数据进行有放回的随机抽样
        X_sample = resample(X, replace=True, n_samples=X.shape[0], random_state=i)
        
        # 在引导样本上做预测
        preds = model.predict(X_sample)
        bootstrap_preds[:, i] = preds
    
    return bootstrap_preds

#可视化
def lasso_pred_plt(X_test,y_test):
    LASSO_saved = joblib.load('./mHHS改变值_best_lasso_model_5.pkl')
    lasso_bootstrap_preds = bootstrap_prediction(LASSO_saved, X_test)

    # 计算每个预测点的平均值和95%置信区间
    lasso_y_pred_mean = lasso_bootstrap_preds.mean(axis=1)
    lasso_y_pred_lower = np.percentile(lasso_bootstrap_preds, 2.5, axis=1)
    lasso_y_pred_upper = np.percentile(lasso_bootstrap_preds, 97.5, axis=1)

    # 结果整合到DataFrame
    lasso_y_pred_df = pd.DataFrame({
        'y_pred_mean': lasso_y_pred_mean,
        'y_pred_lower': lasso_y_pred_lower,
        'y_pred_upper': lasso_y_pred_upper
    })

    # 将真实值和预测值（含置信区间）合并
    lasso_ytrue_and_ypred_with_ci = pd.concat([y_test.reset_index(drop=True), lasso_y_pred_df], axis=1)
    lasso_ytrue_and_ypred_with_ci

    # 可视化预测结果
    # 绘制图形
    plt.figure(figsize=(12, 6))

    # 绘制置信区间
    plt.fill_between(range(len(lasso_ytrue_and_ypred_with_ci)), lasso_ytrue_and_ypred_with_ci['y_pred_lower'], lasso_ytrue_and_ypred_with_ci['y_pred_upper'], color='skyblue', alpha=0.4, label='95% Confidence Interval')

    # 绘制点估计值
    plt.plot(lasso_ytrue_and_ypred_with_ci['y_pred_mean'], color='blue', alpha=0.7, label='Predicted (Mean)')

    # 绘制实际值
    plt.plot(lasso_ytrue_and_ypred_with_ci[y_test.name], color='red', alpha=0.7, label='Actual')

    # 添加图例
    plt.legend()

    # 添加标题和轴标签
    plt.title('Comparison of Actual and Predicted Y with Confidence Interval')
    plt.xlabel('Sample Index')
    plt.ylabel('Y Value')

    # 展示图形
    plt.show()

# %%
for col in y_col:
    print(col)
    Y = y_data[col]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y,test_size=0.2, random_state=0)


    # 加载模型
    filename = fit_lasso_model(X_train, y_train,y_name=col)
    print(filename)
    lasso_model = joblib.load(filename)

    # 获取模型的系数和截距
    coefficients = lasso_model.coef_
    intercept = lasso_model.intercept_

    # 构建方程的字符串表示
    features = [f'x{i+1}' for i in range(len(coefficients))]
    terms = [f'{coefficients[i]:.3f}*{features[i]}' for i in range(len(coefficients))]
    equation = "mHHS = " + f'{intercept:.3f} + ' + ' + '.join(terms)

    # 打印方程
    print(f"{col}_LASSO Regression Equation:")
    print(equation)

        
    # 加载最优LASSO模型
    best_lasso_model = joblib.load(filename)

    y_pred = best_lasso_model.predict(X_test)

    # 计算MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"MSE: {mse}")

    # 计算MAE
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {mae}")

    # 计算R2
    r2 = r2_score(y_test, y_pred)
    print(f"R2: {r2}")

    # 可视化效果
    lasso_pred_plt(X_test,y_test)

    
    

# %% [markdown]
# # 很显然,直接进行回归效果很差,所以下面尝试进行分类

# %% [markdown]
# # 2. 基于MCID的分类

# %%
imputed_data.columns

# %%
y_col =['mHHS_MCID','ADL_MCID',	'iHOT12_MCID','VAS_MCID']
 
## 必须注意删除“'followup随访时间'”，因为这个变量在理论上不会对患者康复、手术效果产生直接影响，本次的预测就是围绕MCID进行

mHHS_data = imputed_data[['年龄','股骨头直径','出现症状至手术时间月','术前α角','术前mHHS','mHHS_MCID']]
ADL_data = imputed_data[['年龄', '性别男1女2', '身高', '体重', '股骨头直径','术前α角', '术前LCEA', '术前mHHS', '术前ADL','ADL_MCID']]
iHOT12_data = imputed_data[['年龄', '体重','股骨头直径','出现症状至手术时间月','术前α角',  '术前mHHS', '术前ADL', '术前iHOT12','iHOT12_MCID',]]
VAS_data = imputed_data[['年龄','体重', '股骨头直径', '关节间隙宽度','出现症状至手术时间月','术前α角', '术前LCEA', '术前mHHS', '术前ADL', '术前iHOT12','VAS_MCID']]

# %%

#  SMOTE使得样本平衡
from imblearn.over_sampling import SMOTE  # SMOTE上采样,把正负样本变成一样的数量,增加负样本
sm = SMOTE(random_state=1)  # 目前而言随机数种子越小越好。但是warning:随机数种子千万不要太大，否则运行时间很长，
datalist = [mHHS_data,ADL_data,iHOT12_data,VAS_data ]
datanamelist = ['mHHS_data','ADL_data','iHOT12_data','VAS_data']
for i in range(4):
    dataset = datalist[i]
    dataname = datanamelist[i]
    X = dataset[dataset.columns[:-1]]
    y = dataset[dataset.columns[-1]]
    X, y = sm.fit_resample(X, y.astype('int'))  # 记得是否降维X还是X_tran
    X = pd.DataFrame(data=X, columns=dataset.columns[:-1])  # 由于X类型发生变化，进行还原df
    balanced_data = pd.concat([X, y], axis=1)
    print('当前数据集shape为', balanced_data.shape)  # 查看结果
    balanced_data.to_excel(f'./{dataname}_balanced.xlsx')

# %%
mHHS_data = pd.read_excel('./mHHS_data_balanced.xlsx')
mHHS_data.drop(columns='Unnamed: 0',inplace=True)
mHHS_data

# %%

ADL_data  = pd.read_excel('./ADL_data_balanced.xlsx')
ADL_data.drop(columns='Unnamed: 0',inplace=True)
iHOT12_data = pd.read_excel('./iHOT12_data_balanced.xlsx')
iHOT12_data.drop(columns='Unnamed: 0',inplace=True)
VAS_data = pd.read_excel('./VAS_data_balanced.xlsx')
VAS_data.drop(columns='Unnamed: 0',inplace=True)

# %% [markdown]
# ### 2.1 模型的训练和评估

# %% [markdown]
# ### 2.1.1 定义模型评价指标

# %%
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import sklearn
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate, KFold, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from lifelines.utils import concordance_index

# 自定义ERR2评分函数（二类错误）
def err2_scorer(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0,1] / (cm[0,1] + cm[0,0])


## c_index
def calculate_c_index(y_true, y_pred):
    """
    Calculates the C-index (Concordance Index) for a given set of true binary labels and predicted probabilities.

    Args:
        y_true (np.ndarray): An array of true binary labels, where `y_true[i]` is either 0 or 1 for the i-th subject.
        y_pred (np.ndarray): An array of predicted probabilities, where `y_pred[i]` is the predicted probability of label 1 for the i-th subject.

    Returns:
        float: The calculated C-index value, which ranges from 0 to 1, with higher values indicating better model performance.
    """
    return concordance_index(y_true, y_pred)


## brier_score
def calculate_brier_score(y_true, y_pred):
    """
    Calculates the Brier Score for a given set of true binary labels and predicted probabilities.

    Args:
        y_true (np.ndarray): An array of true binary labels, where `y_true[i]` is either 0 or 1 for the i-th subject.
        y_pred (np.ndarray): An array of predicted probabilities, where `y_pred[i]` is the predicted probability of label 1 for the i-th subject.

    Returns:
        float: The calculated Brier Score value, which ranges from 0 to 1, with lower values indicating better model performance.
    """
    brier_score = np.mean((y_true - y_pred) ** 2)
    return brier_score

# 基于交叉验证的结果评价模型性能的函数——evaluate_model_with_cv
def evaluate_model_with_cv(X, y, estimator, title, n_splits=10, random_state=12):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # scoring = {
    #     'accuracy': 'accuracy',
    #     'f1_macro': 'f1_macro',
    #     'precision_macro': 'precision_macro',
    #     'recall_macro': 'recall_macro',
    #     'ERR2': make_scorer(err2_scorer)  # 使用make_scorer包装自定义评分函数
    # }

    scoring = {
       'accuracy': 'accuracy',
       'f1_macro': 'f1_macro',
       'precision_macro': 'precision_macro',
       'recall_macro': 'recall_macro',
       'ERR2': make_scorer(err2_scorer),
       'c_index': make_scorer(calculate_c_index),
       'brier_score': make_scorer(calculate_brier_score)
   }
    
    scores = cross_validate(estimator, X, y, cv=cv, scoring=scoring, return_train_score=False)
    results = {metric: (np.mean(scores[metric]), np.std(scores[metric])) for metric in scores if metric.startswith('test_')}
    
    return results


# %% [markdown]
# ### 2.1.2 进行参数寻优(模型训练并保存最佳模型)

# %%
# data
datalist = [mHHS_data,ADL_data,iHOT12_data,VAS_data ]
datanamelist = ['mHHS_data','ADL_data','iHOT12_data','VAS_data']
estimator_name = ['LR', 'SVM', 'RF']  # 模型排序：LR SVM  RF

# %%

## LR

for i in range(4):
    dataset = datalist[i]
    dataname = datanamelist[i]
    print(f'{dataname} result is as follows')
    X = dataset[dataset.columns[:-1]]
    y = dataset[dataset.columns[-1]]

    # Parameter grid for Logistic Regression
    param_grid_lr = {'C': [0.01, 0.1, 1, 10, 20, 30,50,100], 'solver': ['liblinear', 'saga'],}

    # Create the model
    lr_model = LogisticRegression()

    # Perform grid search
    grid_search_lr = GridSearchCV(lr_model, param_grid_lr, cv=10, scoring='accuracy', n_jobs=-1)
    grid_search_lr.fit(X, y)
    print(f"在{dataname}数据集上,LR最优超参数组合为：", grid_search_lr.best_params_)

    # Save the best model
    joblib.dump(grid_search_lr.best_estimator_, f'{dataname}_LR.pkl')


# %%
## SVM

for i in range(4):
    dataset = datalist[i]
    dataname = datanamelist[i]
    print(f'{dataname} result is as follows' )
    X = dataset[dataset.columns[:-1]]
    y = dataset[dataset.columns[-1]]


    # Parameter grid for SVM
    param_grid_svm = {'C': [1, 10],  'gamma': ['scale'],  'kernel': ['rbf'] }

    # Create the model
    svm_model = SVC(probability=True)

    # Perform grid search
    grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=10, scoring='accuracy', n_jobs=-1)
    grid_search_svm.fit(X, y)
    print(f"在{dataname}数据集上,SVM最优超参数组合为：", grid_search_svm.best_params_)

    # Save the best model
    joblib.dump(grid_search_svm.best_estimator_, f'{dataname}_SVM.pkl')


# %%

# RF
for i in range(4):
    dataset = datalist[i]
    dataname = datanamelist[i]
    print(f'{dataname} result is as follows')
    X = dataset[dataset.columns[:-1]]
    y = dataset[dataset.columns[-1]]

    # Parameter grid for Random Forest
    param_grid_rf = {
        'n_estimators': [50,100, 150, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1,2, 4] }

    # Create the model
    rf_model = RandomForestClassifier()

    # Perform grid search
    grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=10, scoring='accuracy', n_jobs=-1)
    grid_search_rf.fit(X, y)
    print(f"在{dataname}数据集上,RF最优超参数组合为：", grid_search_rf.best_params_)

    # Save the best model
    joblib.dump(grid_search_rf.best_estimator_, f'{dataname}_RF.pkl')


# %% [markdown]
# ### 2.1.3 输出最佳模型的性能评价并进行比较

# %%
print('--------------------The results are as follows----------------------------', '\n'*2)
datalist = [mHHS_data,ADL_data,iHOT12_data,VAS_data ]
datanamelist = ['mHHS_data','ADL_data','iHOT12_data','VAS_data']
estimator_name = ['LR', 'SVM', 'RF']  # 模型排序：LR SVM  RF

# 初始化存储结果的列表
results_list = []

# 循环遍历数据集和模型
for i, dataset in enumerate(datalist):
    dataname = datanamelist[i]
    X = dataset[dataset.columns[:-1]]
    y = dataset[dataset.columns[-1]]
    
    for model_name in estimator_name:
        filename = f'./{dataname}_{model_name}.pkl'
        estimator = joblib.load(filename)  # 加载模型
        
        # 使用evaluate_model_with_cv评估模型
        results = evaluate_model_with_cv(X, y, estimator, f"{dataname}_{model_name}")
        
        # 将结果添加到列表中
        for metric, (mean, std) in results.items():
            results_list.append({
                'Dataset': dataname,
                'Model': model_name,
                'Metric': metric,
                'Mean': mean,
                'STD': std
            })

# 将结果列表转换为DataFrame
results_df = pd.DataFrame(results_list)

# 将DataFrame保存为Excel文件
results_df.to_excel("model_evaluation_results0407.xlsx", index=False)

print(results_df)
        
    


# %%
# 画出ROC曲线
# 循环遍历数据集和模型
estimator_name = ['LR', 'SVM', 'RF']
print(estimator_name)


import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
plt.rcParams['font.family'] = 'Times New Roman'



for i, dataset in enumerate(datalist):
    dataname = datanamelist[i]
    X = dataset[dataset.columns[:-1]]
    y = dataset[dataset.columns[-1]]
    
    plt.figure(figsize=(16, 16))
    plt.plot([0, 1], [0, 1], linestyle='--',linewidth=2.5)

    for i in range(3):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=12) 
        estimator_model = estimator_name[i]
        filename = f'D:/business-code/sport_predict/{dataname}_{estimator_model}.pkl'
        model_lr = joblib.load(filename)
        model_lr.fit(X_train, y_train)
        ytest_predict_prob = model_lr.predict_proba(X_test)
        # print(type(ytest_predict_prob), ytest_predict_prob)
        ytest_predict_prob = ytest_predict_prob[:, 1]
        #  print(type(ytest_predict_prob), ytest_predict_prob)
        # calculate AUC
        test_auc = roc_auc_score(y_test, ytest_predict_prob, average='micro',multi_class='ovo')
        # calculate roc curve
        test_fpr, test_tpr, test_thresholds = roc_curve(y_test, ytest_predict_prob)
        # plot the roc curve for the model
        plt.plot(test_fpr, test_tpr, label='{}-AUC: {:.6f}'.format(estimator_name[i], test_auc),linewidth=3.5)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlabel('False Positive Rate', fontsize=30)
        plt.ylabel('True Positive Rate', fontsize=30)
        plt.grid(True)  # 添加网格线以增强可读性
        plt.legend(loc='lower right',fontsize=40)
    plt.savefig("%s AUC-curve.png" % dataname)
    plt.show()
    plt.close()


print('All is ok,you can have a rest\n'*3)




# %%
#导入全部所需要的库
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
from shapash.data.data_loader import data_loading
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import shap
shap.initjs()   # notebook环境下，加载用于可视化的JS代码
from shapash.explainer.smart_explainer import SmartExplainer
import joblib 
import shap
from sklearn.ensemble import RandomForestClassifier

# %%
mHHS_data = pd.read_excel('./mHHS_data_balanced.xlsx')
mHHS_data.drop(columns='Unnamed: 0',inplace=True)
ADL_data  = pd.read_excel('./ADL_data_balanced.xlsx')
ADL_data.drop(columns='Unnamed: 0',inplace=True)
iHOT12_data = pd.read_excel('./iHOT12_data_balanced.xlsx')
iHOT12_data.drop(columns='Unnamed: 0',inplace=True)
VAS_data = pd.read_excel('./VAS_data_balanced.xlsx')
VAS_data.drop(columns='Unnamed: 0',inplace=True)

# %%
X_test = ADL_data[ADL_data.columns[:-1]]

## load data
filename = f'./ADL_data_RF.pkl'
print(filename)
RF_saved = joblib.load(filename)
# 使用SHAP解释模型
explainer = shap.Explainer(RF_saved)
shap_values = explainer(X_test)
# 获取特征重要性的平均绝对值
shap_sum = np.abs(shap_values.values).mean(axis=0)
importance_df = pd.DataFrame([X_test.columns.tolist(), shap_sum.tolist()]).T
importance_df.columns = ['Feature', 'SHAP Importance']
importance_df = importance_df.sort_values('SHAP Importance', ascending=False)

# 打印特征重要性排序
print(importance_df)



# %%
datalist = [mHHS_data,ADL_data,iHOT12_data,VAS_data ]
datanamelist = ['mHHS_data','ADL_data','iHOT12_data','VAS_data']


for i in range(4):
    dataset = datalist[i]
    dataname = datanamelist[i]
    X_test = dataset[dataset.columns[:-1]]  # 假设您已经正确加载了每个数据集

    list1 = list(X_test.columns)
    print('目前所有的特征的数量是:',len(list1))
    print('目前所有特征的名称:',list1)
    list2 = list(range(0,len(list1))) 
    print(len(list2),list2)
    fdata = pd.DataFrame({'a': list2, 'fea': list1})
    X_dict = dict(zip(fdata['a'], fdata['fea']))
    print(X_dict)

    # 加载保存的模型
    filename = f'./{dataname}_RF.pkl'
    print(f'{filename}的SHAP特征重要性解释如下')
    RF_saved = joblib.load(filename)

    # 使用SHAP解释模型
    explainer = shap.TreeExplainer(RF_saved)
    shap_values = explainer(X_test)
    # 获取特征重要性的平均绝对值
    shap_sum = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame([X_test.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ['Feature', 'SHAP Importance']
    importance_df = importance_df.sort_values('SHAP Importance', ascending=False)

    # 打印特征重要性排序
    print(importance_df)
    importance_df.to_excel(f'./{dataname}_importance.xlsx')





# %%
#导入全部所需要的库
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
from shapash.data.data_loader import data_loading
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import shap
shap.initjs()   # notebook环境下，加载用于可视化的JS代码
from shapash.explainer.smart_explainer import SmartExplainer
import joblib 
import shap
from sklearn.ensemble import RandomForestRegressor
import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
plt.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

import warnings
warnings.filterwarnings('ignore', category=UserWarning)



mHHS_data = pd.read_excel('./mHHS_data_balanced.xlsx')
mHHS_data.drop(columns='Unnamed: 0',inplace=True)
ADL_data  = pd.read_excel('./ADL_data_balanced.xlsx')
ADL_data.drop(columns='Unnamed: 0',inplace=True)
iHOT12_data = pd.read_excel('./iHOT12_data_balanced.xlsx')
iHOT12_data.drop(columns='Unnamed: 0',inplace=True)
VAS_data = pd.read_excel('./VAS_data_balanced.xlsx')
VAS_data.drop(columns='Unnamed: 0',inplace=True)



# %%

X = mHHS_data[mHHS_data.columns[:-1]]  
y = mHHS_data[mHHS_data.columns[-1]]


# 重新训练一个RF然后解释
RF_model =  RandomForestClassifier(min_samples_leaf= 1, min_samples_split= 2, n_estimators= 50)
RF_model.fit(X, y)
y_pre = pd.DataFrame({'y-pre': RF_model.predict(X)}, index=X.index)


# 使用SHAP解释模型
explainer = shap.TreeExplainer(RF_model)
shap_values = explainer(X)
print('原始SHAP:',shap_values.shape)
# 只要正样本的SHAP值
shap_values = shap_values[..., 0]
print('修改后SHAP:',shap_values.shape)
# print(shap_values)
# print(type(shap_values))



# bar plot
shap.plots.bar(shap_values,max_display=5)  #  max_display=5
plt.show()


# # Summary Plot (概要图，红色条表示正向影响（即特征值增加时，模型预测值也会增加），而蓝色条表示负向影响。)
shap.summary_plot(shap_values, X, ) # plot_type="beeswarm"
shap.summary_plot(shap_values, features=X, feature_names=X.columns, plot_type="bar")

plt.show()


# heatmap（热力图，颜色越暖表示正向影响越大，颜色越冷表示负向影响越大。）
shap.plots.heatmap(shap_values)

plt.show()


# %% [markdown]
# ADL

# %%

X = ADL_data[ADL_data.columns[:-1]]  
y = ADL_data[ADL_data.columns[-1]]


# 重新训练一个RF然后解释=
RF_model =  RandomForestClassifier(max_depth=30, min_samples_leaf=1, min_samples_split= 5, n_estimators=20)
RF_model.fit(X, y)
y_pre = pd.DataFrame({'y-pre': RF_model.predict(X)}, index=X.index)


# 使用SHAP解释模型
explainer = shap.TreeExplainer(RF_model)
shap_values = explainer(X)
print('原始SHAP:',shap_values.shape)
# 只要正样本的SHAP值
shap_values = shap_values[..., 0]
print('修改后SHAP:',shap_values.shape)
# print(shap_values)
# print(type(shap_values))



# bar plot
shap.plots.bar(shap_values,max_display=5)  #  max_display=5
plt.show()




# # Summary Plot (概要图，红色条表示正向影响（即特征值增加时，模型预测值也会增加），而蓝色条表示负向影响。)
shap.summary_plot(shap_values, X, ) # plot_type="beeswarm"
shap.summary_plot(shap_values, features=X, feature_names=X.columns, plot_type="bar")


plt.show()


# heatmap（热力图，颜色越暖表示正向影响越大，颜色越冷表示负向影响越大。）
shap.plots.heatmap(shap_values)

plt.show()



# %% [markdown]
# iHOT12

# %%

X = iHOT12_data[iHOT12_data.columns[:-1]]  
y = iHOT12_data[iHOT12_data.columns[-1]]


# 重新训练一个RF然后解释=
RF_model =  RandomForestClassifier(max_depth=20, min_samples_leaf= 2, min_samples_split=5, n_estimators= 50)
RF_model.fit(X, y)
y_pre = pd.DataFrame({'y-pre': RF_model.predict(X)}, index=X.index)


# 使用SHAP解释模型
explainer = shap.TreeExplainer(RF_model)
shap_values = explainer(X)
print('原始SHAP:',shap_values.shape)
# 只要正样本的SHAP值
shap_values = shap_values[..., 0]
print('修改后SHAP:',shap_values.shape)
# print(shap_values)
# print(type(shap_values))



# bar plot
shap.plots.bar(shap_values,max_display=5)  #  max_display=5
plt.show()




# # Summary Plot (概要图，红色条表示正向影响（即特征值增加时，模型预测值也会增加），而蓝色条表示负向影响。)
shap.summary_plot(shap_values, X, ) # plot_type="beeswarm"
shap.summary_plot(shap_values, features=X, feature_names=X.columns, plot_type="bar")


plt.show()


# heatmap（热力图，颜色越暖表示正向影响越大，颜色越冷表示负向影响越大。）
shap.plots.heatmap(shap_values)

plt.show()



# %%

X = VAS_data[VAS_data.columns[:-1]]  
y = VAS_data[VAS_data.columns[-1]]


# 重新训练一个RF然后解释=
RF_model =  RandomForestClassifier(max_depth= 20, min_samples_leaf=1, min_samples_split=2, n_estimators=200)
RF_model.fit(X, y)
y_pre = pd.DataFrame({'y-pre': RF_model.predict(X)}, index=X.index)


# 使用SHAP解释模型
explainer = shap.TreeExplainer(RF_model)
shap_values = explainer(X)
print('原始SHAP:',shap_values.shape)
# 只要正样本的SHAP值
shap_values = shap_values[..., 0]
print('修改后SHAP:',shap_values.shape)
# print(shap_values)
# print(type(shap_values))



# bar plot
shap.plots.bar(shap_values,max_display=5)  #  max_display=5
plt.show()





# # Summary Plot (概要图，红色条表示正向影响（即特征值增加时，模型预测值也会增加），而蓝色条表示负向影响。)
shap.summary_plot(shap_values, X, ) # plot_type="beeswarm"
shap.summary_plot(shap_values, features=X, feature_names=X.columns, plot_type="bar")


plt.show()


# heatmap（热力图，颜色越暖表示正向影响越大，颜色越冷表示负向影响越大。）
shap.plots.heatmap(shap_values)

plt.show()



# %%



