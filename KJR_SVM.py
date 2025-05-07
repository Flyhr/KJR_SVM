import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ----------------------------
# 1. 数据加载与预处理
# ----------------------------
data = pd.read_csv('integrated_data.csv')

# 删除不需要的字段
drop_columns = ['id', 'name', 'predict_result', 'doctor_result', 'create_time', 'update_time']
data.drop(columns=drop_columns, inplace=True)

# 填充症状描述（decri）的缺失值为空字符串
data['decri'] = data['decri'].fillna('')

# 将类别字段转换为数值
data['gender'] = data['gender'].map({'男': 1, '女': 0})
data['smoking'] = data['smoking'].map({'是': 1, '否': 0})
data['swelling'] = data['swelling'].map({'是': 1, '否': 0})

# 删除含缺失值的样本
data.dropna(inplace=True)

# 确保目标列 pain 为整型（康复等级：0, 1, 2）
data['pain'] = data['pain'].astype(int)

print("处理后的数据预览：")
print(data.head())

# ----------------------------
# 2. 特征选择与数据集划分
# ----------------------------
numeric_cols = ['weight', 'height', 'gender', 'age', 'smoking', 'temp', 'temp_cut',
                'pulse', 'sbp', 'dbp', 'swelling', 'knee', 'step']
text_col = 'decri'
X = data[numeric_cols + [text_col]]
y = data['pain']

# 使用 train_test_split 随机划分数据集，保持类别分布
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"总数据量: {len(data)}, 训练集: {len(X_train)}, 测试集: {len(X_test)}")

# ----------------------------
# 3. 构建特征预处理流水线
# ----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('txt', TfidfVectorizer(max_features=100), text_col)
    ]
)

# ----------------------------
# 4. 构建 SVM 模型流水线与超参数调优
# ----------------------------
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('svc', SVC(kernel='rbf', random_state=42))
])

# 设置需要搜索的超参数空间
param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 'auto', 0.1, 0.01]
}

# 使用 GridSearchCV 进行交叉验证参数搜索
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print("最佳参数:", grid.best_params_)

# 使用最佳模型评估测试集
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("模型准确率:", accuracy)
print("分类报告:")
print(classification_report(y_test, y_pred))
print("训练集 pain 分布:")
print(y_train.value_counts(normalize=True))
print("测试集 pain 分布:")
print(y_test.value_counts(normalize=True))
print("验证样本预测结果:", y_pred.tolist())

# 可选：使用交叉验证评估整体模型表现
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print("5折交叉验证平均准确率:", cv_scores.mean())

# ----------------------------
# 5. 保存训练好的模型流水线到文件
# ----------------------------
joblib.dump(best_model, "svm_pipeline.pkl")
print("模型已保存到 svm_pipeline.pkl")
