import numpy as np

class FeatureWeightedAttentionForest(AttentionForest):
    # ... (остальной код)

    def optimize_weights(self, X, y) -> 'FeatureWeightedAttentionForest':
        assert self.forest is not None, "Need to fit before weights optimization"
        y = self._preprocess_target(y)
        dynamic_weights, dynamic_y = self._get_dynamic_weights_y(X)

        if dynamic_y.shape[2] == 1:
            dynamic_y = dynamic_y[..., 0]

        w_init = np.ones(self.forest.n_estimators)
        feature_weights_init = np.ones(X.shape[1])
        if self.params.eps is not None:
            static_weights_init = np.ones(self.forest.n_estimators) / self.forest.n_estimators

        learning_rate = 1e-4
        weight_decay = 1e-5
        num_epochs = 100

        for epoch in range(num_epochs):
            for i in range(len(X)):
                x_batch = X[i:i+1]
                y_batch = y[i:i+1]

                # Рассчитываем предсказания
                new_dyn_weights = -0.5 * np.linalg.norm(dynamic_weights / feature_weights_init, 2, axis=-1) ** 2.0
                alphas = new_dyn_weights * np.abs(w_init)
                alphas_softmax = np.exp(alphas) / np.sum(np.exp(alphas), axis=1, keepdims=True)
                if self.params.eps is not None:
                    static_softmax = np.exp(static_weights_init) / np.sum(np.exp(static_weights_init))
                    mixed_weights = (1.0 - self.params.eps) * alphas_softmax + self.params.eps * static_softmax
                else:
                    mixed_weights = alphas_softmax

                predictions = np.sum(np.multiply(mixed_weights, dynamic_y[i:i+1]), axis=1)

                # Рассчитываем потери
                loss_terms = predictions - y_batch
                if self.params.loss_ord == 1:
                    loss = np.sum(np.abs(loss_terms))
                elif self.params.loss_ord == 2:
                    loss = np.sum(loss_terms ** 2)
                else:
                    raise ValueError(f'Wrong loss order: {self.params.loss_ord}')

                # Обновляем веса
                gradient_w = np.sum(np.multiply(alphas_softmax, dynamic_y[i:i+1]), axis=1)
                gradient_feature_weights = np.sum(np.multiply(alphas_softmax * w_init[:, np.newaxis], dynamic_weights / feature_weights_init[:, np.newaxis]), axis=0)

                w_init -= learning_rate * gradient_w
                feature_weights_init -= learning_rate * gradient_feature_weights

                if self.params.eps is not None:
                    gradient_static_weights = np.sum(np.multiply(mixed_weights - alphas_softmax, dynamic_y[i:i+1]), axis=1)
                    static_weights_init -= learning_rate * gradient_static_weights

        self.tree_weights = w_init.copy()
        self.feature_weights = feature_weights_init.copy()
        if self.params.eps is not None:
            self.static_weights = np.exp(static_weights_init) / np.sum(np.exp(static_weights_init))

        return self


import numpy as np

class FeatureWeightedAttentionForest(AttentionForest):
    # ... (остальной код)

    def optimize_weights(self, X, y) -> 'FeatureWeightedAttentionForest':
        assert self.forest is not None, "Need to fit before weights optimization"
        y = self._preprocess_target(y)
        dynamic_weights, dynamic_y = self._get_dynamic_weights_y(X)

        if dynamic_y.shape[2] == 1:
            dynamic_y = dynamic_y[..., 0]

        w_init = np.ones(self.forest.n_estimators)
        feature_weights_init = np.ones(X.shape[1])
        if self.params.eps is not None:
            static_weights_init = np.ones(self.forest.n_estimators) / self.forest.n_estimators

        learning_rate = 1e-4
        weight_decay = 1e-5
        num_epochs = 100

        for epoch in range(num_epochs):
            for i in range(len(X)):
                x_batch = X[i:i+1]
                y_batch = y[i:i+1]

                # Рассчитываем предсказания
                new_dyn_weights = -0.5 * np.linalg.norm(dynamic_weights / feature_weights_init, 2, axis=-1) ** 2.0
                alphas = new_dyn_weights * np.abs(w_init)
                alphas_softmax = np.exp(alphas - np.max(alphas)) / np.sum(np.exp(alphas - np.max(alphas)), axis=1, keepdims=True)
                if self.params.eps is not None:
                    static_softmax = np.exp(static_weights_init - np.max(static_weights_init)) / np.sum(np.exp(static_weights_init - np.max(static_weights_init)))
                    mixed_weights = (1.0 - self.params.eps) * alphas_softmax + self.params.eps * static_softmax
                else:
                    mixed_weights = alphas_softmax

                predictions = np.sum(np.multiply(mixed_weights, dynamic_y[i:i+1]), axis=1)

                # Рассчитываем потери
                loss_terms = predictions - y_batch
                if self.params.loss_ord == 1:
                    loss = np.sum(np.abs(loss_terms))
                elif self.params.loss_ord == 2:
                    loss = np.sum(loss_terms ** 2)
                else:
                    raise ValueError(f'Wrong loss order: {self.params.loss_ord}')

                # Обновляем веса
                gradient_w = np.sum(np.multiply(alphas_softmax, dynamic_y[i:i+1]), axis=1)
                gradient_feature_weights = np.sum(np.multiply(alphas_softmax * w_init[:, np.newaxis], dynamic_weights / feature_weights_init[:, np.newaxis]), axis=0)

                w_init -= learning_rate * gradient_w
                feature_weights_init -= learning_rate * gradient_feature_weights

                if self.params.eps is not None:
                    gradient_static_weights = np.sum(np.multiply(mixed_weights - alphas_softmax, dynamic_y[i:i+1]), axis=1)
                    static_weights_init -= learning_rate * gradient_static_weights

        self.tree_weights = w_init.copy()
        self.feature_weights = feature_weights_init.copy()
        if self.params.eps is not None:
            self.static_weights = np.exp(static_weights_init - np.max(static_weights_init)) / np.sum(np.exp(static_weights_init - np.max(static_weights_init)))

        return self
