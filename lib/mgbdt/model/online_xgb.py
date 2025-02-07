import numpy as np
from xgboost.core import DMatrix
from xgboost.training import train
from xgboost.sklearn import XGBModel, _objective_decorator


class OnlineXGB(XGBModel):
    def fit_increment(self, X, y, num_boost_round=1, params=None):  #params是XGBModel中存储参数的字典 此处默认的params是None
        trainDmatrix = DMatrix(X, label=y, nthread=self.n_jobs, missing=self.missing)
        extra_params = params           #extra_params为输入的params值 若未输入则为None 
        params = self.get_xgb_params()  #params重新定义为XGBModel中的params 
        
        #如果输入了params则用输入值 没有输入则用XGBModel中的params 
        if extra_params is not None:  
            for k, v in extra_params.items():
                params[k] = v
        #原本此处没有if语句 实际运行时报错KeyError: 'n_estimators' (对字典中某个key进行操作时 如该key不存在 就会报这个错)
        #加上if语句进行key是否存在的判断
        if "n_estimators" in params:
            params.pop("n_estimators")  #n_estimators基学习器的数量 即迭代次数

        if callable(self.objective):
            obj = _objective_decorator(self.objective)
            params["objective"] = "reg:linear"
        else:
            obj = None

        #_Booster存储训练好的XGBModel
        if self._Booster is None:  
            self._Booster = train(
                    params=params,
                    dtrain=trainDmatrix,
                    num_boost_round=num_boost_round,
                    obj=obj)
        else:
            self._Booster = train(
                    params=params,
                    dtrain=trainDmatrix,
                    num_boost_round=num_boost_round,
                    obj=obj,
                    xgb_model=self._Booster)
        return self

    def predict(self, X):
        if self._Booster is None:
            return np.full((X.shape[0],), self.base_score)
        return super(OnlineXGB, self).predict(X)
