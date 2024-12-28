# 商务智能2024年秋大作业-KKBox音乐偏好分析及推荐算法设计

本报告围绕音乐偏好分析及推荐算法的设计展开，以Kaggle平台上KKBOX音乐推荐挑战赛的数据集为基础，完成了“用户行为预测”和“音乐推荐”两大任务。报告首先对数据集进行了基本的异常清洗和缺失处理，接着进行了特征工程，使用随机森林算法对原始特征的影响程度进行了排序，并用SVD算法提取了用户-歌曲对的隐藏特征。对于这些特征，报告对比了XGBoost和LightGBM两个模型的训练效果，最终分别达到了72%和76%的准确率，在Kaggle平台历史数据中名列前茅。最后，报告还利用Gradio平台搭建了一个小型的音乐推荐应用，供用户使用。

# 使用方法

安装依赖：`pip install -r ./requirements.txt` 

报告代码以记事本形式编写，执行对应功能只需运行相应的记事本。

1. 数据预处理`data_process.ipynb`

2. 数据可视化和音乐偏好分析`visualization.ipynb`

3. 特征工程`feqture_engineering.ipynb`

4. 模型训练`train_XGBoost.ipynb` `train_LightGBM.ipynb` 

5. Gradio音乐推荐应用`python recommendation.py` 