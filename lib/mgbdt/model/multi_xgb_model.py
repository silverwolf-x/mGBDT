import numpy as np
from joblib import Parallel, delayed

from .online_xgb import OnlineXGB

#MultiXGB算法 多输出XGB 输出的结果可以是多维 
class MultiXGBModel:
    def __init__(self, input_size, output_size, learning_rate, max_depth=5, num_boost_round=1, force_no_parallel=False, **kwargs):
        """
        model: (XGBoost)
        """
        self.input_size = input_size                #输入样本的特征数
        self.output_size = output_size              #输出结果的维度
        self.learning_rate = learning_rate          #学习率
        self.max_depth = max_depth                  #决策树深度(层数)
        self.num_boost_round = num_boost_round      #每个GBDT里的决策树数量(集成次数)
        self.force_no_parallel = force_no_parallel  #大概是跟多线程有关 False时多个model同时训练 True时model一个接一个训练
        self.models = []                            #设输出维度为n 大模型是n输出GBDT 由n个单输出GBDT组成
        for i in range(self.output_size):  
            single_model = OnlineXGB(max_depth=max_depth, silent=True, n_jobs=-1, learning_rate=learning_rate, **kwargs)
            self.models.append(single_model)        #大模型 = n输出GBDT = n个单输出GBDT

    def __repr__(self):  #获取模型
        return "MultiXGBModel(input_size={}, output_size={}, learning_rate={:.3f}, max_depth={}, num_boost_round={})".format(
                self.input_size, self.output_size, self.learning_rate, self.max_depth, self.num_boost_round)     #{:.3f}保留3位小数

    def __call__(self, *args, **kwargs):  #__call__使MultiXGBModel类的实例像函数一样可以被调用
        return self.predict(*args, **kwargs)

    def fit(self, X, y, num_boost_round=None, params=None):  #定义训练函数
        assert X.shape[1] == self.input_size  # X.shape[0]行数=样本数   X.shape[1]列数=特征数
        if self.force_no_parallel:
            self._fit_serial(X, y, num_boost_round, params)
        else:
            self._fit_parallel(X, y, num_boost_round, params)

    def predict(self, X):  #定义预测函数
        assert X.shape[1] == self.input_size  #assert断言 如果后面表达式为False会报错
        out = self._predict_serial(X)
        if not self.force_no_parallel and (X.shape[0] <= 10000 or X.shape[1] <= 10):
            out = self._predict_parallel(X)
        else:
            out = self._predict_serial(X)
        return out

    def _fit_serial(self, X, y, num_boost_round, params):  #一个接一个训练模型 每个模型训练时使用cpu所有线程
        if num_boost_round is None:
            num_boost_round = self.num_boost_round
        for i in range(self.output_size):
            self.models[i].n_jobs = -1  #使用CPU所有的线程数进行运算，起到并行加速的作用
        for i in range(self.output_size):
            self.models[i].fit_increment(X, y[:, i], num_boost_round=num_boost_round, params=None)

    def _fit_parallel(self, X, y, num_boost_round, params):  #同时训练所有模型 每个模型训练时使用cpu的1个线程
        if num_boost_round is None:
            num_boost_round = self.num_boost_round
        for i in range(self.output_size):
            self.models[i].n_jobs = 1  #使用CPU的1个线程进行运算
            
        #Parallel对象会创建一个进程池，以便在多进程中执行每一个列表项。
        #函数delayed是一个创建元组(function, args, kwargs)
        #以下相当于将fit_increment函数分到多个进程上执行 每个模型使用1个进程
        Parallel(n_jobs=-1, verbose=False, backend="threading")(
                delayed(model.fit_increment)(X, y[:, i], num_boost_round=num_boost_round, params=None)
                for i, model in enumerate(self.models))

    def _predict_serial(self, X):  #模型一个接一个进行预测 每个模型进行预测时使用cpu所有线程
        for i in range(self.output_size):
            self.models[i].n_jobs = -1
        pred = np.empty((X.shape[0], self.output_size), dtype=np.float64)
        for i in range(self.output_size):
            pred[:, i] = self.models[i].predict(X)
        return pred

    def _predict_parallel(self, X):  #所有模型同时进行预测 每个模型进行预测时使用cpu的1个线程
        for i in range(self.output_size):
            self.models[i].n_jobs = 1
        pred = Parallel(n_jobs=-1, verbose=False, backend="threading")(
                delayed(model.predict)(X)
                for i, model in enumerate(self.models))
        pred = np.asarray(pred, dtype=np.float64).T
        return pred
