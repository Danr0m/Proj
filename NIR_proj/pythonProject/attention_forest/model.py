import logging
import numpy as np
import cvxpy
from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, Tuple
from sklearn.model_selection import train_test_split
from .forests import *
from scipy.special import softmax as _softmax
import cvxpy as cp
import scipy
from time import time
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize


class ClfRegHot:
    @abstractmethod
    def fit(self, X, y) -> 'ClfRegHot':
        pass

    @abstractmethod
    def refit(self, X, y) -> 'ClfRegHot':
        """Fit to the new dataset with warm start."""
        pass

    @abstractmethod
    def optimize_weights(self, X, y) -> 'ClfRegHot':
        """Optimize weights for the new dataset."""
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass

    @abstractmethod
    def predict_original(self, X) -> np.ndarray:
        pass


class AFParams(NamedTuple):
    """Parameters of Attention Forest."""
    kind: ForestKind
    task: TaskType
    loss_ord: int = 2  # loss order
    eps: Optional[int] = None
    discount: Optional[float] = None
    forest: dict = {}


class EAFParams(AFParams):
    """Parameters of Epsilon-Attention Forest."""

    # eps: float = 0.1
    # tau: float = 1.0
    def __new__(cls, *args, eps: float = 0.1, tau: float = 1.0, **kwargs):
        self = super(EAFParams, cls).__new__(cls, *args, eps=eps, **kwargs)
        self.tau = tau
        return self


class FWAFParams(AFParams):
    """Parameters of Feature-Weighted Attention Forest."""

    def __new__(cls, *args, no_temp: bool = False, **kwargs):
        self = super(FWAFParams, cls).__new__(cls, *args, **kwargs)
        self.no_temp = no_temp
        return self


class LeafData(NamedTuple):
    xs: np.ndarray
    y: np.ndarray


def _prepare_leaf_data_fast(xs, y, leaf_ids, estimators, multiplier=1.0):
    # Определяется максимальный идентификатор листа среди всех оценщиков (деревьев) в лесу.
    # Это необходимо для создания массива результатов, который будет иметь достаточный размер для хранения данных о каждом листе.
    max_leaf_id = max(map(lambda e: e.tree_.node_count, estimators))
    # Определяется длина оси y.
    # Если y является одномерным массивом, то y_len будет равно 1. В противном случае, если y является двумерным массивом, y_len будет равно числу столбцов.
    y_len = 1 if y.ndim == 1 else y.shape[1]
    # result_x предназначен для хранения предварительно обработанных данных о входных признаках (xs).
    # result_y предназначен для хранения предварительно обработанных данных о целевых переменных (y).
    # Оба массива имеют форму (len(estimators), max_leaf_id + 1, xs.shape[1]) для result_x и (len(estimators), max_leaf_id + 1, y_len) для result_y.
    result_x = np.full((len(estimators), max_leaf_id + 1, xs.shape[1]), np.nan, dtype=np.float32)
    result_y = np.full((len(estimators), max_leaf_id + 1, y_len), np.nan, dtype=np.float32)
    # по всем деревьям в лесу
    for tree_id in range(len(estimators)):
        # по каждому листу в дереве
        for leaf_id in range(estimators[tree_id].tree_.node_count + 1):
            # Для каждого листа выполняется маскирование входных данных xs и целевых переменных y с использованием leaf_ids, чтобы выбрать только те данные, которые относятся к текущему листу.
            mask = (leaf_ids[:, tree_id] == leaf_id)
            masked_xs = xs[mask]
            masked_y = y[mask]
            # если хотя бы один образец относится к текущему листу (mask.any()), вычисляется среднее значение признаков masked_xs и целевых переменных masked_y.
            if mask.any():
                result_x[tree_id, leaf_id] = masked_xs.mean(axis=0)
                result_y[tree_id, leaf_id] = masked_y.mean(axis=0) * multiplier
                print("Tree id:" )
                print(tree_id)
                print("leaf_id:")
                print(leaf_id)
                print("x:")
                print(masked_xs.shape)
                print(masked_xs)
                print("to:")
                print(result_x[tree_id, leaf_id])
                print(result_x[tree_id, leaf_id].shape)
                print("y:")
                print(masked_y.shape)
                print(masked_y)
                print("to:")
                print(result_y[tree_id, leaf_id].shape)
                print(result_y[tree_id, leaf_id])
    return result_x, result_y

def _prepare_leaf_data_cl(xs, y, leaf_ids, estimators, multiplier=1.0):
    np.set_printoptions(suppress=True)
    # Определяется максимальный идентификатор листа среди всех оценщиков (деревьев) в лесу.
    # Это необходимо для создания массива результатов, который будет иметь достаточный размер для хранения данных о каждом листе.
    max_leaf_id = max(map(lambda e: e.tree_.node_count, estimators))
    # Определяется длина оси y.
    # Если y является одномерным массивом, то y_len будет равно 1. В противном случае, если y является двумерным массивом, y_len будет равно числу столбцов.
    y_len = 1 if y.ndim == 1 else y.shape[1]
    # result_x предназначен для хранения предварительно обработанных данных о входных признаках (xs).
    # result_y предназначен для хранения предварительно обработанных данных о целевых переменных (y).
    # Оба массива имеют форму (len(estimators), max_leaf_id + 1, xs.shape[1]) для result_x и (len(estimators), max_leaf_id + 1, y_len) для result_y.
    result_x = np.full((len(estimators), max_leaf_id + 1, xs.shape[1]), np.nan, dtype=np.float32)
    result_y = np.full((len(estimators), max_leaf_id + 1, y_len), np.nan, dtype=np.float32)
    # по всем деревьям в лесу
    for tree_id in range(len(estimators)):
        # по каждому листу в дереве
        for leaf_id in range(estimators[tree_id].tree_.node_count + 1):
            # Для каждого листа выполняется маскирование входных данных xs и целевых переменных y с использованием leaf_ids, чтобы выбрать только те данные, которые относятся к текущему листу.
            mask = (leaf_ids[:, tree_id] == leaf_id)
            masked_xs = xs[mask]
            masked_y = y[mask]
            # если хотя бы один образец относится к текущему листу (mask.any()), вычисляется среднее значение признаков masked_xs и целевых переменных masked_y.
            if mask.any():
                #y = количество x соответствующих классу y деленное на общее кол-во x
                result_x[tree_id, leaf_id] = masked_xs.mean(axis=0) #A(x)


                result_y[tree_id, leaf_id] = masked_y.sum(axis=0)/len(result_x[tree_id, leaf_id]) #(B(x))

                # print("Tree id:" )
                # print(tree_id)
                # print("leaf_id:")
                # print(leaf_id)
                # print("---------")
                # print(masked_y.sum(axis=0))
                # print(len(result_x[tree_id, leaf_id]))
                # print("---------")
                # print("x:")
                # print(masked_xs.shape)
                # print(masked_xs)
                # print("to:")
                # print(result_x[tree_id, leaf_id])
                # print(result_x[tree_id, leaf_id].shape)
                # print("y:")
                # print(masked_y.shape)
                # print(masked_y)
                # print("to:")
                # print(result_y[tree_id, leaf_id].shape)
                # print(result_y[tree_id, leaf_id])
    return result_x, result_y


def _convert_labels_to_probas(y, encoder=None):
    # имеет ли y две оси и если вторая ось (количество столбцов) больше или равна 2.
    if y.ndim == 2 and y.shape[1] >= 2:
        return y, encoder
    if encoder is None:
        encoder = OneHotEncoder()
        y = encoder.fit_transform(y.reshape((-1, 1)))
    else:
        y = encoder.transform(y.reshape((-1, 1)))
    return y, encoder


class AttentionForest(ClfRegHot):
    def __init__(self, params: AFParams):
        self.params = params
        self.forest = None
        self._after_init()

    def _after_init(self):
        self.onehot_encoder = None

    def _make_leaf_data(self, leaf_id, tree_leaf_ids, xs, y):
        # маска, чтобы выбрать только те образцы, которые принадлежат листу с идентификатором leaf_id
        mask = (tree_leaf_ids == leaf_id)
        masked_xs = xs[mask]
        masked_y = y[mask]
        return LeafData(xs=masked_xs, y=masked_y)

    def _preprocess_target(self, y):
        if self.params.task == TaskType.CLASSIFICATION:
            y, self.onehot_encoder = _convert_labels_to_probas(y, self.onehot_encoder)
        return y

    def fit(self, X, y) -> 'AttentionForest':
        # Определяется класс леса (forest_cls), который будет использоваться в зависимости от типа леса (kind) и типа задачи (task), определенных в параметрах модели.
        forest_cls = FORESTS[ForestType(self.params.kind, self.params.task)]
        # Создается экземпляр леса (self.forest) с использованием определенного класса леса (forest_cls) и параметров леса (**self.params.forest).
        self.forest = forest_cls(**self.params.forest)
        self.forest.fit(X, y)
        self.training_xs = X.copy()
        self.training_y = self._preprocess_target(y.copy())
        # store leaf id for each point in X
        self.training_leaf_ids = self.forest.apply(self.training_xs)
        # print("Random forest apply time:", end_time - start_time)
        # make a tree-leaf-points correspondence
        # print("Generating leaves data")
        start_time = time()
        # multiplier = self.forest.learning_rate if self.params.kind.need_add_init() else 1.0
        multiplier = 1.0
        # Проверяется, есть ли у объекта self.forest атрибут или метод get_leaf_data
        # Если нет, то используется функция _prepare_leaf_data_fast для подготовки данных о листьях на основе полученных параметров.
        if hasattr(self.forest, 'get_leaf_data'):
            self.leaf_data_x, self.leaf_data_y = self.forest.get_leaf_data()
        else:
            self.leaf_data_x, self.leaf_data_y = _prepare_leaf_data_cl(
                self.training_xs,
                self.training_y,
                self.training_leaf_ids,
                self.forest.estimators_,
                multiplier=multiplier
            )
        end_time = time()
        # print("Leaf generation time:", end_time - start_time)
        # Инициализируются атрибуты tree_weights и static_weights массивами из единиц
        self.tree_weights = np.ones(self.forest.n_estimators)
        self.static_weights = np.ones(self.forest.n_estimators) / self.forest.n_estimators
        return self

    def optimize_weights(self, X, y_orig) -> 'AttentionForest':
        raise NotImplementedError("Use EpsAttentionForest or FeatureWeightedAttentionForest instead.")

    def refit(self, X, y) -> 'AttentionForest':
        assert self.forest is not None, "Need to fit before refit"
        # update leaves?
        raise NotImplementedError()
        # find tree weights
        self.optimize_weights(X, y)
        return self

    def _get_dynamic_weights_y(self, X) -> Tuple[np.ndarray, np.ndarray]:
        #Вычисляются идентификаторы листьев для каждого образца входных данных X с использованием метода apply леса
        leaf_ids = self.forest.apply(X)
        # пустые списки для хранения динамических весов и целевых переменных
        all_dynamic_weights = []
        all_y = []
        # итерация по каждому образцу cur_x и соответствующим ему идентификаторам листьев cur_leaf_ids
        for cur_x, cur_leaf_ids in zip(X, leaf_ids):
            tree_dynamic_weights = []
            tree_dynamic_y = []
            #итерация по каждому дереву в лесу и соответствующему ему идентификатору листа cur_leaf_id.
            for cur_tree_id, cur_leaf_id in enumerate(cur_leaf_ids):
                # Получаем средние значения признаков и целевых переменных для данного листа cur_leaf_id в дереве cur_tree_id из данных о листьях self.leaf_data_x и self.leaf_data_y.
                leaf_mean_x = self.leaf_data_x[cur_tree_id][cur_leaf_id]
                leaf_mean_y = self.leaf_data_y[cur_tree_id][cur_leaf_id]
                # вычисляем d(x, A_k(x)) для данного образца cur_x и листа cur_leaf_id в дереве cur_tree_id на основе расстояния между образцом и средним значением признаков листа.
                tree_dynamic_weight = -0.5 * np.linalg.norm(cur_x - leaf_mean_x, 2) ** 2.0
                # Проверяется, задано ли снижение веса (discount) в параметрах модели
                if self.params.discount is not None:
                    # Если задано снижение веса, динамический вес дополнительно умножается на значение снижения веса в степени cur_tree_id.
                    tree_dynamic_weight *= self.params.discount ** cur_tree_id
                tree_dynamic_weights.append(tree_dynamic_weight)
                tree_dynamic_y.append(leaf_mean_y)
            tree_dynamic_weights = np.array(tree_dynamic_weights)
            tree_dynamic_y = np.array(tree_dynamic_y)
            all_dynamic_weights.append(tree_dynamic_weights)
            all_y.append(tree_dynamic_y)
        all_dynamic_weights = np.array(all_dynamic_weights)
        all_y = np.array(all_y)
        return all_dynamic_weights, all_y

    def predict(self, X) -> np.ndarray:
        assert self.forest is not None, "Need to fit before predict"
        # Вычисляются динамические веса и целевые переменные для всех образцов входных данных X с помощью метода _get_dynamic_weights_y
        all_dynamic_weights, all_y = self._get_dynamic_weights_y(X)
        # Динамические веса нормализуются с использованием функции softmax.
        # self.tree_weights используется для взвешивания динамических весов каждого дерева.
        # axis=1 указывает, что нормализация должна выполняться по деревьям.
        weights = _softmax(all_dynamic_weights * self.tree_weights[np.newaxis], axis=1)
        if self.params.eps is not None:
            # Вычисляются смешанные веса (mixed_weights), учитывая параметр eps, если он был задан
            mixed_weights = (1.0 - self.params.eps) * weights + self.params.eps * self.static_weights
        else:
            mixed_weights = weights
        #Добавляется новая ось
        mixed_weights = mixed_weights[..., np.newaxis]
        #Вычисляются предсказания путем взвешенной суммы целевых переменных all_y  по деревьям  с использованием смешанных весов mixed_weights.
        predictions = np.sum(mixed_weights * all_y, axis=1)
        if self.params.kind.need_add_init():
            predictions += self.forest.init_.predict(X)
        return predictions

    def predict_proba(self, X):
        assert self.forest is not None, "Need to fit before predict"
        all_dynamic_weights, all_y = self._get_dynamic_weights_y(X)
        weights = _softmax(all_dynamic_weights * self.tree_weights[np.newaxis], axis=1)
        if self.params.eps is not None:
            mixed_weights = (1.0 - self.params.eps) * weights + self.params.eps * self.static_weights
        else:
            mixed_weights = weights
        mixed_weights = mixed_weights[..., np.newaxis]
        predictions = np.sum(mixed_weights * all_y, axis=1)

        # Преобразуем значения в вероятности, используя softmax
        # Каждое значение будет преобразовано в вероятность для каждого класса
        if predictions.ndim == 1:
            # Для случая бинарной классификации
            predictions_proba = np.column_stack((1 - predictions, predictions))
        else:
            # Для многоклассовой классификации
            predictions_proba = _softmax(predictions, axis=1)

        return predictions_proba

    def predict_original(self, X):
        if self.params.task == TaskType.REGRESSION:
            return self.forest.predict(X)
        elif self.params.task == TaskType.CLASSIFICATION:
            return self.forest.predict_proba(X)
        raise ValueError(f'Unsupported task type in predict_original: "{self.params.task}"')


class EpsAttentionForest(AttentionForest):
    def __init__(self, params: EAFParams):
        self.params = params
        self.forest = None
        self.w = None
        self._after_init()

    def fit(self, X, y):
        super().fit(X, y)
        self.w = np.ones(self.forest.n_estimators) / self.forest.n_estimators

    def optimize_weights(self, X, y_orig) -> 'EpsAttentionForest':
        assert self.forest is not None, "Need to fit before weights optimization"
        # Получение динамических весов и целевых переменных на основе входных данных X с помощью метода _get_dynamic_weights_y.
        dynamic_weights, dynamic_y = self._get_dynamic_weights_y(X)
        static_weights = cp.Variable((1, self.forest.n_estimators))

        bias = 0.0
        if self.params.kind.need_add_init():
            bias = self.forest.init_.predict(X)
        y = y_orig.copy()
        y -= bias

        if dynamic_y.shape[2] == 1:
            dynamic_y = dynamic_y[..., 0]
            # dynamic_weights = D_k(x_s, tau), static_weights = B_k(x_s)
            mixed_weights = (1.0 - self.params.eps) * dynamic_weights + self.params.eps * static_weights
        else:
            # y0: y0t0_0 y0t0_1 ... y0t0_d | y0t1_0 y0t1_1 ... y0t1_d | ... | y0tT_0 ... y0tT_d
            # loss = sum_i sum_j (sum_k yitk_j - yi_j)
            # dynamic_y = dynamic_y.reshape((dynamic_y.shape[0], -1))
            # swap 1 and 2 axes and merge "sample" and "feature" axes
            n_trees = dynamic_y.shape[1]
            n_outs = dynamic_y.shape[2]
            dynamic_y = np.transpose(dynamic_y, (0, 2, 1)).reshape((-1, dynamic_y.shape[1]))
            y = y.reshape((-1))
            # dynamic_weights shape: (n_samples, n_trees)
            # repeat dynamic weights for each output
            dynamic_weights = np.tile(dynamic_weights[:, np.newaxis, :], (1, n_outs, 1)).reshape((-1, n_trees))
            # dynamic_weights = D_k(x_s, tau), static_weights = B_k(x_s)
            mixed_weights = (1.0 - self.params.eps) * dynamic_weights + self.params.eps * static_weights #alpha = mixed_weights
            y, self.onehot_encoder = _convert_labels_to_probas(y, self.onehot_encoder)
            y = y.toarray().ravel()
            #print("Shapes:", mixed_weights.shape, dynamic_y.shape)
        loss_terms = cp.sum(cp.multiply(mixed_weights, dynamic_y), axis=1) - y
        if self.params.loss_ord == 1:
            min_obj = cp.sum(cp.abs(loss_terms))
        elif self.params.loss_ord == 2:
            min_obj = cp.sum_squares(loss_terms)
        else:
            raise ValueError(f'Wrong loss order: {self.params.loss_ord}')
        problem = cp.Problem(cp.Minimize(min_obj),
                             [
                                 static_weights >= 0,
                                 cp.sum(static_weights, axis=1) == 1
                             ]
                             )

        try:
            loss_value = problem.solve()
        except Exception as ex:
            logging.warning(f"Solver error: {ex}")

        if static_weights.value is None:
            logging.info(f"Can't solve problem with OSQP. Trying another solver...")
            loss_value = problem.solve(solver=cvxpy.SCS)

        if static_weights.value is None:
            logging.warn(f"Weights optimization error (eps={self.params.eps}). Using default values.")
        else:
            self.w = static_weights.value.copy().reshape((-1,))
        return self

    def _get_dynamic_weights_y(self, X) -> Tuple[np.ndarray, np.ndarray]:
        leaf_ids = self.forest.apply(X)
        all_dynamic_weights = []
        all_y = []
        for cur_x, cur_leaf_ids in zip(X, leaf_ids):
            tree_dynamic_weights = []
            tree_dynamic_y = []
            for cur_tree_id, cur_leaf_id in enumerate(cur_leaf_ids):
                leaf_mean_x = self.leaf_data_x[cur_tree_id][cur_leaf_id]
                leaf_mean_y = self.leaf_data_y[cur_tree_id][cur_leaf_id]
                tree_dynamic_weight = -0.5 * np.linalg.norm(cur_x - leaf_mean_x, 2) ** 2.0
                if self.params.discount is not None:
                    tree_dynamic_weight *= self.params.discount ** cur_tree_id
                tree_dynamic_weights.append(tree_dynamic_weight)
                tree_dynamic_y.append(leaf_mean_y)
            tree_dynamic_weights = _softmax(np.array(tree_dynamic_weights) * self.params.tau)
            tree_dynamic_y = np.array(tree_dynamic_y)
            all_dynamic_weights.append(tree_dynamic_weights)
            all_y.append(tree_dynamic_y)
        all_dynamic_weights = np.array(all_dynamic_weights)
        all_y = np.array(all_y)
        return all_dynamic_weights, all_y

    def predict(self, X) -> np.ndarray:
        assert self.forest is not None, "Need to fit before predict"
        all_dynamic_weights, all_y = self._get_dynamic_weights_y(X)
        mixed_weights = (1.0 - self.params.eps) * all_dynamic_weights + self.params.eps * self.w
        mixed_weights = mixed_weights[..., np.newaxis]
        predictions = np.sum(mixed_weights * all_y, axis=1)
        if self.params.kind.need_add_init():
            predictions += self.forest.init_.predict(X)[:, np.newaxis]
        return predictions

    def predict_proba(self, X):
        assert self.forest is not None, "Need to fit before predict_proba"
        all_dynamic_weights, all_y = self._get_dynamic_weights_y(X)
        all_dynamic_weights = -0.5 * np.linalg.norm(all_dynamic_weights / self.feature_weights, 2, axis=-1) ** 2.0
        weights = _softmax(all_dynamic_weights * self.tree_weights[np.newaxis], axis=1)
        if self.params.eps is not None:
            mixed_weights = (1.0 - self.params.eps) * weights + self.params.eps * self.static_weights
        else:
            mixed_weights = weights
        mixed_weights = mixed_weights[..., np.newaxis]
        predictions = np.sum(mixed_weights * all_y, axis=1)

        # Преобразуем значения в вероятности, используя softmax
        # Каждое значение будет преобразовано в вероятность для каждого класса
        if predictions.ndim == 1:
            # Для случая бинарной классификации
            predictions_proba = np.column_stack((1 - predictions, predictions))
        else:
            # Для многоклассовой классификации
            predictions_proba = _softmax(predictions, axis=1)

        return predictions_proba


class FeatureWeightedAttentionForest(AttentionForest):
    def __init__(self, params: FWAFParams):
        self.params = params
        self.forest = None
        self._after_init()

    def fit(self, X, y) -> 'AttentionForest':
        super().fit(X, y)
        n_features = X.shape[1]
        self.feature_weights = np.ones(n_features)
        return self

    def optimize_weights(self, X, y) -> 'AttentionForest':
        assert self.forest is not None, "Need to fit before weights optimization"
        y = self._preprocess_target(y)
        dynamic_weights, dynamic_y = self._get_dynamic_weights_y(X) #x - A(x), p(x)

        if dynamic_y.shape[2] == 1:
            dynamic_y = dynamic_y[..., 0]

        w_init = np.ones(self.forest.n_estimators)
        n_features = X.shape[1]
        feature_weights_init = np.ones(n_features)
        if self.params.eps is not None:
            static_weights_init = np.ones(self.forest.n_estimators) / self.forest.n_estimators



        def _model(cur_feature_weights, cur_w, cur_static_weights):
            new_dyn_weights = -0.5 * np.linalg.norm(dynamic_weights / cur_feature_weights, 2, axis=-1) ** 2.0
            alphas = new_dyn_weights * np.abs(cur_w)
            alphas_softmax = _softmax(alphas, axis=1)
            if self.params.eps is not None:
                static_softmax = _softmax(cur_static_weights, axis=0)
                mixed_weights = (1.0 - self.params.eps) * alphas_softmax + self.params.eps * static_softmax
            else:
                mixed_weights = alphas_softmax
            if dynamic_y.ndim == 3:
                mixed_weights = mixed_weights[..., np.newaxis]
            return np.sum(
                np.multiply(mixed_weights, dynamic_y),
                axis=1
            )

        def _loss(cur_preds, y_true):
            loss_terms = cur_preds - y_true
            if self.params.loss_ord == 1:
                loss = np.sum(np.abs(loss_terms))
            elif self.params.loss_ord == 2:
                if loss_terms.ndim == 2:
                    loss = np.sum(np.linalg.norm(loss_terms, 2, axis=1) ** 2)
                else:
                    loss = np.sum(loss_terms ** 2)
            else:
                raise ValueError(f'Wrong loss order: {self.params.loss_ord}')
            return loss

        # print("Dynamic_y", dynamic_y.shape)
        opt_params = [w_init, feature_weights_init]

        if self.params.eps is not None:
            opt_params.append(static_weights_init)
        if self.params.no_temp:
            # don't optimize `w`
            params_to_optimize = opt_params[1:]
        else:
            params_to_optimize = opt_params
        params_to_optimize_lens = list(map(len, params_to_optimize))

        def _min_fn(cur_merged_weights):
            # split weights
            cur_weights = [
                cur_merged_weights[prev_cum_len:prev_cum_len + cur_len]
                for prev_cum_len, cur_len in
                zip([0] + list(np.cumsum(params_to_optimize_lens))[:-1], params_to_optimize_lens)
            ]

            cur_static_weights = cur_weights[0]

            if self.params.no_temp:
                cur_tree_weights = w_init
                idx = 0
            else:
                cur_tree_weights = cur_weights[0]
                idx = 1
            cur_feature_weights = cur_weights[idx]
            if self.params.eps is not None:
                cur_static_weights = cur_weights[idx + 1]
            cur_preds = _model(cur_feature_weights, cur_tree_weights, cur_static_weights)
            loss = _loss(cur_preds, y)
            return loss

        params_to_optimize = np.concatenate(params_to_optimize, axis=0)
        #print(params_to_optimize.shape)
        result = scipy.optimize.minimize(_min_fn, params_to_optimize, method='L-BFGS-B', jac=False)
        optimal_weights = [
            result.x[prev_cum_len:prev_cum_len + cur_len]
            for prev_cum_len, cur_len in
            zip([0] + list(np.cumsum(params_to_optimize_lens))[:-1], params_to_optimize_lens)
        ]

        if self.params.no_temp:
            idx = 0
        else:
            self.tree_weights = optimal_weights[0].copy()
            idx = 1
        self.feature_weights = optimal_weights[idx].copy()
        if self.params.eps is not None:
            self.static_weights = _softmax(optimal_weights[idx + 1].copy())

        return self

    #-------------------
    def optimize_weights_sgd(self, X, y) -> 'AttentionForest':
        assert self.forest is not None, "Need to fit before weights optimization"
        y = self._preprocess_target(y)
        dynamic_weights, dynamic_y = self._get_dynamic_weights_y(X) #dynamic_weights - A()

        if dynamic_y.shape[2] == 1:
            dynamic_y = dynamic_y[..., 0]

        w_init = np.ones(self.forest.n_estimators)
        n_features = X.shape[1]
        feature_weights_init = np.ones(n_features)

        if self.params.eps is not None:
            static_weights_init = np.ones(self.forest.n_estimators) / self.forest.n_estimators

        def _model(fn_dyn_weights, fn_dyn_y, curr_w, curr_feature_weights):
            np.set_printoptions(suppress=True)
            #fn_dyn_weights = x - A_k(x)
            new_dyn_weights = -0.5 * np.linalg.norm(fn_dyn_weights / curr_feature_weights, 2, axis=-1) ** 2.0
            #alphas = alpha(x,A_k(x),v,z)
            alphas = new_dyn_weights * np.abs(curr_w)
            alphas_softmax = _softmax(alphas, axis=1)
            # print("alphas")
            # print(alphas)

            if self.params.eps is not None:
                static_softmax = _softmax(static_weights_init, axis=0)
                mixed_weights = (1.0 - self.params.eps) * alphas_softmax + self.params.eps * static_softmax
            else:
                mixed_weights = alphas_softmax
            if dynamic_y.ndim == 3:
                mixed_weights = mixed_weights[..., np.newaxis]
            # print("summa")
            # print(np.sum(
            #     np.multiply(mixed_weights, fn_dyn_y),
            #     axis=1
            # ))
            #sum alpha()p_k(x)
            return np.sum(
                np.multiply(mixed_weights, fn_dyn_y),
                axis=1
            )

        def _loss(cur_preds, y_true):
            #L(p(x),h,v,z)
            loss_terms = cur_preds - y_true
            # print("loss_terms")
            # print(loss_terms)
            if self.params.loss_ord == 1:
                loss = np.sum(np.abs(loss_terms))
            elif self.params.loss_ord == 2:
                if loss_terms.ndim == 2:
                    loss = np.sum(np.linalg.norm(loss_terms, 2, axis=1) ** 2)
                else:
                    loss = np.sum(loss_terms ** 2)
            else:
                raise ValueError(f'Wrong loss order: {self.params.loss_ord}')
            # print("loss")
            # print(loss)
            return loss

        print("Dynamic_y", dynamic_y.shape)

        opt_params = [w_init, feature_weights_init]

        if self.params.eps is not None:
            opt_params.append(static_weights_init)
        if self.params.no_temp:
            # don't optimize `w`
            params_to_optimize = opt_params[1:]
        else:
            params_to_optimize = opt_params
        params_to_optimize_lens = list(map(len, params_to_optimize))
        print("params_to_optimize_lens", params_to_optimize_lens)

        #стохастический градиентный спуск
        def sgd_opt(model_func, loss_func, fn_opt_params, inputs, y, n_epochs=100, batch_size=1, learning_rate=0.01):

            num_samples = y.shape[0]
            #num_features = len(params_to_optimize_lens)

            def _calculate_gradients(params):
                gradients_S = [[0] * len(sublist) for sublist in fn_opt_params]
                for batch_idx in range(0, num_samples, batch_size):
                    batch_inputs = [inp[batch_idx:batch_idx + batch_size] for inp in inputs]
                    batch_y = y[batch_idx:batch_idx + batch_size]
                    param_grad = [[0] * len(sublist) for sublist in fn_opt_params]

                    # for i in range(num_features):
                    #
                    #     perturbation = np.zeros_like(params[i])
                    #     perturbation[i] = 1e-6
                    #
                    #     params_pos = params.copy()
                    #     params_neg = params.copy()
                    #
                    #     sum = params_pos[i] + perturbation[i]
                    #
                    #     params_pos[i] = sum
                    #     params_neg[i] = params_neg[i] - perturbation[i]
                    #
                    #
                    #     loss_pos = loss_func(model_func(*batch_inputs, *params_pos), batch_y)
                    #     loss_neg = loss_func(model_func(*batch_inputs, *params_neg), batch_y)
                    #
                    #     new_array = [(loss_pos - loss_neg) / (2 * 1e-6)] * len(param_grad[i])
                    #     param_grad[i] = new_array

                    perturbation = [[0] * len(sublist) for sublist in params]
                    perturbation = [[0.5] * len(sublist) for sublist in perturbation]
                    params_pos = params.copy()
                    params_neg = params.copy()
                    params_pos = [[x + y for x, y in zip(sublist1, sublist2)] for sublist1, sublist2 in zip(params_pos, perturbation)]
                    params_neg = [[x - y for x, y in zip(sublist1, sublist2)] for sublist1, sublist2 in zip(params_neg, perturbation)]
                    loss_pos = loss_func(model_func(*batch_inputs, *params_pos), batch_y)
                    #print("loss_pos")
                    # print(loss_pos)
                    loss_neg = loss_func(model_func(*batch_inputs, *params_neg), batch_y)
                    #print("loss_neg")
                    # print(loss_neg)
                    param_grad = [[(loss_pos - loss_neg) / (2 * 1e-6)] * len(sublist) for sublist in param_grad]

                    gradients_S = [[x + y for x, y in zip(sublist1, sublist2)] for sublist1, sublist2 in zip(gradients_S, param_grad)]
                gradients_S = [[x / (num_samples / batch_size) for x in sublist] for sublist in gradients_S]
                return gradients_S

            for epoch in range(n_epochs):
                gradients = _calculate_gradients(fn_opt_params)
                gradients = [[x * learning_rate for x in sublist] for sublist in gradients]
                fn_opt_params = [[x - y for x, y in zip(sublist1, sublist2)] for sublist1, sublist2 in zip(fn_opt_params, gradients)]

            return fn_opt_params


        optimal_weights = sgd_opt(
            _model,
            _loss,
            params_to_optimize,
            [  # inputs
                dynamic_weights,
                dynamic_y,
            ],
            y,
            n_epochs=100,
            batch_size=4,
        )


        self.tree_weights = optimal_weights[0].copy()
        self.feature_weights = optimal_weights[1].copy()

        if self.params.eps is not None:
            self.static_weights = _softmax(optimal_weights[2].copy())

        return self
    #-------------------
    def _get_dynamic_weights_y(self, X) -> Tuple[np.ndarray, np.ndarray]:
        np.set_printoptions(suppress=True)
        leaf_ids = self.forest.apply(X)
        all_dynamic_weights = []
        all_y = []
        for cur_x, cur_leaf_ids in zip(X, leaf_ids):
            tree_dynamic_weights = []
            tree_dynamic_y = []
            for cur_tree_id, cur_leaf_id in enumerate(cur_leaf_ids):
                leaf_mean_x = self.leaf_data_x[cur_tree_id][cur_leaf_id]
                leaf_mean_y = self.leaf_data_y[cur_tree_id][cur_leaf_id]
                # tree_dynamic_weight = -0.5 * np.linalg.norm(cur_x - leaf_mean_x, 2) ** 2.0
                # NOTE: here `tree_dynamic_weight` are not actually weights
                tree_dynamic_weight = (cur_x - leaf_mean_x)#x - A(x)
                tree_dynamic_weights.append(tree_dynamic_weight)
                tree_dynamic_y.append(leaf_mean_y)#p(x)
            tree_dynamic_weights = np.array(tree_dynamic_weights)#A_k(x)
            tree_dynamic_y = np.array(tree_dynamic_y)
            all_dynamic_weights.append(tree_dynamic_weights)
            all_y.append(tree_dynamic_y)#p(x)
        all_dynamic_weights = np.array(all_dynamic_weights)
        all_y = np.array(all_y)
        return all_dynamic_weights, all_y

    def predict(self, X) -> np.ndarray:
        assert self.forest is not None, "Need to fit before predict"
        all_dynamic_weights, all_y = self._get_dynamic_weights_y(X)
        print("  all_dynamic_weights shape:", all_dynamic_weights.shape, self.feature_weights.shape)
        all_dynamic_weights = -0.5 * np.linalg.norm(all_dynamic_weights / self.feature_weights, 2, axis=-1) ** 2.0
        weights = _softmax(all_dynamic_weights * self.tree_weights[np.newaxis], axis=1)
        if self.params.eps is not None:
            mixed_weights = (1.0 - self.params.eps) * weights + self.params.eps * self.static_weights
        else:
            mixed_weights = weights
        mixed_weights = mixed_weights[..., np.newaxis]
        predictions = np.sum(mixed_weights * all_y, axis=1)
        return predictions

    def predict_proba(self, X):
        assert self.forest is not None, "Need to fit before predict_proba"
        all_dynamic_weights, all_y = self._get_dynamic_weights_y(X)
        all_dynamic_weights = -0.5 * np.linalg.norm(all_dynamic_weights / self.feature_weights, 2, axis=-1) ** 2.0
        #print(self.tree_weights)
        weights = _softmax(all_dynamic_weights * self.tree_weights, axis=1)
        if self.params.eps is not None:
            mixed_weights = (1.0 - self.params.eps) * weights + self.params.eps * self.static_weights
        else:
            mixed_weights = weights
        mixed_weights = mixed_weights[..., np.newaxis]
        predictions = np.sum(mixed_weights * all_y, axis=1)

        # Преобразуем значения в вероятности, используя softmax
        # Каждое значение будет преобразовано в вероятность для каждого класса
        if predictions.ndim == 1:
            # Для случая бинарной классификации
            predictions_proba = np.column_stack((1 - predictions, predictions))
        else:
            # Для многоклассовой классификации
            predictions_proba = _softmax(predictions, axis=1)

        return predictions_proba

    def predict_original(self, X):
        if self.params.task == TaskType.REGRESSION:
            return self.forest.predict(X)
        elif self.params.task == TaskType.CLASSIFICATION:
            return self.forest.predict_proba(X)
        raise ValueError(f'Unsupported task type in predict_original: "{self.params.task}"')


def fit_forest_split(X, y, params: AFParams, pre_size: float = 0.75, seed: Optional[int] = None):
    X_pre, X_post, y_pre, y_post = train_test_split(X, y, train_size=pre_size, random_state=seed)
    model = AttentionForest(params)
    model.fit(X_pre, y_pre)
    # model.refit(X_post, y_post)
    model.optimize_weights(X_post, y_post)
    return model