import numpy as np
import six
from termcolor import colored

from mgbdt.layer import BPLayer, TPLayer
from mgbdt.loss import get_loss
from mgbdt.model import MultiXGBModel
from mgbdt.utils.log_utils import logger


class MGBDT:
    def __init__(self, loss=None, target_lr=0.1, epsilon=0.3, verbose=False):
        """
        Attributes
        ----------
        layers: list
            layers[0] is just a stub for convinience
            layers[1] is the first layer
            layers[M] is the last layer (M is the number of layers)
        """
        if loss is not None and isinstance(loss, six.string_types):
            self.loss = get_loss(loss)
        else:
            self.loss = loss
        self.target_lr = target_lr
        self.epsilon = epsilon
        self.verbose = verbose  #verbose是日志显示，有三个参数可选择，分别为0不输出、1输出带进度条的日志信息、2输出不带进度条的
        self.layers = [None]

    @property
    def is_last_layer_bp(self):
        return isinstance(self.layers[-1], BPLayer)

    @property
    ## 返回层数(第一层是凑数的所以要-1)
    def n_layers(self):
        return len(self.layers) - 1

    ## 建立新层
    def add_layer(self, layer_type, *args, **kwargs):
        if layer_type == "tp_layer":
            layer_class = TPLayer
        elif layer_type == "bp_layer":
            layer_class = BPLayer
        else:
            raise ValueError()
        layer = layer_class(*args, **kwargs)
        self.layers.append(layer)

    ## 返回某一层输出结果(默认返回最后一层输出)
    def forward(self, X, n_layers=None):
        M = self.n_layers if n_layers is None else n_layers
        layers = self.layers
        out = X
        for i in range(1, M + 1):
            out = layers[i].forward(out)
        return out

    ## 返回每一层输出结果
    def get_hiddens(self, X, n_layers=None):
        """
        Return
        ------
        H: ndarray, shape = [M + 1, ]
            M represent the number of layers
            H[0] is the inputs
            H[1] is the outputs of the first layer
            H[M] is the outputs of the last layer (the outputs)
        """
        M = self.n_layers if n_layers is None else n_layers
        layers = self.layers
        H = [None for _ in range(M + 1)]
        H[0] = X
        for i in range(1, M + 1):
            H[i] = layers[i].forward(H[i - 1])
        return H

    ## 初始化前向映射F1,F2,F3,...,Fn
    def init(self, X, n_rounds=1, learning_rate=None, max_depth=None, batch=0):
        self.log("[init][start]")
        layers = self.layers
        M = self.n_layers 
        params = {}
        if learning_rate is not None:
            params["learning_rate"] = learning_rate
        if max_depth is not None:
            params["max_depth"] = max_depth
        for i in range(1, M + 1):
            self.log("[init] layer={}".format(i))
            rand_out = np.random.randn(X.shape[0], layers[i].output_size)
            if isinstance(layers[i].F, MultiXGBModel):
                layers[i].F.fit(X, rand_out, num_boost_round=n_rounds, params=params.copy())
            X = layers[i].forward(X)
        self.log("[init][end]")

    ## 建立伪逆映射 即训练每一层的G
    def fit_inverse_mapping(self, X):
        self.log("[fit_inverse_mapping][start] X.shape={}".format(X.shape))
        M = self.n_layers   ## 获取层数-1
        H = self.get_hiddens(X)  ## 获取所有层的输出
        for i in range(2, M + 1):
            layer = self.layers[i]
            if hasattr(layer, "fit_inverse_mapping"):  ## hasattr(object, name)用于判断对象是否包含对应的属性
                layer.fit_inverse_mapping(H[i - 1], epsilon=self.epsilon)
                ## 调用layer类中的方法fit_inverse_mapping(H[i - 1], epsilon=self.epsilon)
                ## 该方法中实际调用G.fit( F(H[i-1]+e), H[i-1]+e ) 即以该层的 输出值为自变量、输入值为因变量 进行拟合 
                ## G是某个模型类的对象 mGBDT算法中G是GBDT
        self.log("[fit_inverse_mapping][end]")

    ## 建立前向映射 即训练每一层的F
    def fit_forward_mapping(self, X, y):
        self.log("[fit_forward_mapping][start] X.shape={}, y.shape={}".format(X.shape, y.shape))
        layers = self.layers
        M = self.n_layers
        # 2.1 Compute hidden units in prediction 计算上一次迭代后隐藏层的输出结果
        self.log("2.1 Compute hidden units")
        H = self.get_hiddens(X)
        # 2.2 Compute the targets 计算上一次迭代后的伪标签 (backward是layer类的函数 tplayer类返回伪逆映射的输出值 bplayer类返回)
        self.log("2.2 Compute the targets")
        Ht = [None for _ in range(M + 1)]
        if self.is_last_layer_bp:   ## 若最后一层是反向传播bp
            Ht[M] = y  ## 则令最后一层伪标签=y
        else:  ## 若最后一层是目标传播tp
            gradient = self.loss.backward(H[M], y)  ## 则计算梯度gradient
            Ht[M] = H[M] - self.target_lr * gradient  ## 最后一层伪标签=输出-lr*梯度
        for i in range(M, 1, -1):
            ## 判断每层的传播方式 (只有最后一层可能是反向传播bp)
            ## 若该层是bp 前一层伪标签 = 对前一层的输出值进行梯度下降
            if isinstance(layers[i], BPLayer):
                assert i == M, "Only last layer can be BackPropogation Layer. i={}".format(i)
                Ht[i - 1] = layers[i].backward(H[i - 1], Ht[i], self.target_lr)
            ## 若该层是tp 前一层伪标签 = 后一层伪逆映射的输出值
            else:
                Ht[i - 1] = layers[i].backward(H[i - 1], Ht[i])
        # 2.3 Training feedward mapping 训练本次迭代的新的前向映射
        self.log("2.3 Training feedward mapping")
        for i in range(1, M + 1):
            if i == 1:
                H = X
            else:
                H = layers[i - 1].forward(H)
            fit_inputs = H
            fit_targets = Ht[i]
            self.log("fit layer={}".format(i))
            layers[i].fit_forward_mapping(fit_inputs, fit_targets)
            # layers[i].fit_forward_mapping(fit_inputs + np.random.randn(*fit_inputs.shape) * self.epsilon, fit_targets)
        self.log("[fit_forward_mapping][end]")
        return H, Ht

    def fit(self, X, y, n_epochs=1, eval_sets=None, batch=0, callback=None, n_eval_epochs=1, eval_metric=None):
        if eval_sets is None:
            eval_sets = ()
        self.log_metric("[epoch={}/{}][train]".format(0, n_epochs), X, y, eval_metric)
        for (x_test, y_test) in eval_sets:
            self.log_metric("[epoch={}/{}][test]".format(0, n_epochs), x_test, y_test, eval_metric)
        batch = min(len(X), batch)
        if batch == 0:
            batch = len(X)
        for _epoch in range(1, n_epochs + 1):
            if batch < len(X):
                perm = np.random.permutation(len(X))
            else:
                perm = list(range(len(X)))
            for si in range(0, len(perm) - batch + 1, batch):
                rand_idx = perm[si: si + batch]
                self.fit_inverse_mapping(X[rand_idx])
            for si in range(0, len(perm) - batch + 1, batch):
                rand_idx = perm[si: si + batch]
                H, Ht = self.fit_forward_mapping(X[rand_idx], y[rand_idx])
            if n_eval_epochs > 0 and _epoch % n_eval_epochs == 0:
                self.log_metric("[epoch={}/{}][train]".format(_epoch, n_epochs), X, y, eval_metric)
                for (x_test, y_test) in eval_sets:
                    self.log_metric("[epoch={}/{}][test]".format(_epoch, n_epochs), x_test, y_test, eval_metric)
            if callback is not None:
                callback(self, _epoch, X, y, eval_sets)
            si += batch

    def mse(self, a, b):
        return np.mean((a - b)**2)

    def log_metric(self, prefix, X, y, eval_metric):
        pred = self.forward(X)
        loss = self.calc_loss(pred, y)
        if eval_metric == "accuracy":
            from sklearn.metrics import accuracy_score
            score = accuracy_score(y, pred.argmax(axis=1)) 
            # //argmax(axis=1)返回每一行最大值的索引，每行取值∈{0,1,...,N-1}
            # //对于N分类问题，y∈{0,1,...,N-1}
            # //设置输出结果为N维，即pred为N维，返回最大值索引，索引的取值就在{0,1,...,N-1}
            # //哪一个位置取值最大，就分入哪一类
        else:
            score = None
        if score is None:
            self.log("{} loss={:.6f}".format(prefix, loss), "green")
        else:
            self.log("{} loss={:.6f}, score={:.6f}".format(prefix, loss, score), "green")

    def calc_loss(self, pred, target):
        try:
            if self.is_last_layer_bp:
                loss = self.layers[-1].calc_loss(pred, target)
            else:
                loss = self.loss.forward(pred, target)
        except Exception:
            try:
                loss = self.mse(pred, target)
            except Exception:
                return np.nan
        return loss

    def log(self, msg, color=None):
        if color is None:
            if self.verbose:
                logger.info(msg)
        else:
            logger.info(colored("{}".format(msg), color))

    def __repr__(self):
        res = "\n(MGBDT STRUCTURE BEGIN)\n"
        res += "loss={}, target_lr={:.3f}, epsilon={:.3f}\n".format(self.loss, self.target_lr, self.epsilon)
        for li, layer in enumerate(self.layers):
            if li == 0:
                continue
            res += "[layer={}] {}\n".format(li, layer)
        res += "(MGBDT STRUCTURE END)"
        return res
