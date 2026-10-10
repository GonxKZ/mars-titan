"""Bosques XGBoost escritos a mano en su formato JSON, sin llamar a `xgb.train`.

Sirven para comprobar y medir rutas de predicción con árboles de forma conocida mientras
el aprendizaje está bloqueado. Rasgos, umbrales y hojas son aleatorios y reproducibles.
"""

import json

import numpy as np


def hand_forest(cuts, features, *, trees=40, depth=4, seed=3, on_cuts=True):
    """Bosque sin ajustar con umbrales en los cortes `(indptr, values)` o entre ellos."""
    import xgboost as xgb

    indptr, cut_values = (np.asarray(part) for part in cuts)
    rng = np.random.default_rng(seed)
    internal, total = 2**depth - 1, 2 ** (depth + 1) - 1
    encoded = []
    for index in range(trees):
        split_indices, conditions = [], []
        for node in range(total):
            if node >= internal:
                split_indices.append(0)
                conditions.append(float(np.float32(rng.normal() * 0.1)))
                continue
            feature = int(rng.integers(features))
            own = cut_values[indptr[feature] : indptr[feature + 1]]
            # El último corte supera el máximo: hist no divide en él.
            position = int(rng.integers(max(1, len(own) - 1)))
            threshold = own[position]
            if not on_cuts and position + 1 < len(own):
                threshold = (own[position] + own[position + 1]) / 2
            split_indices.append(feature)
            conditions.append(float(np.float32(threshold)))
        left = [2 * n + 1 if n < internal else -1 for n in range(total)]
        right = [2 * n + 2 if n < internal else -1 for n in range(total)]
        parents = [2147483647] + [(n - 1) // 2 for n in range(1, total)]
        encoded.append(
            dict(
                base_weights=[c if n >= internal else 0.0 for n, c in enumerate(conditions)],
                categories=[],
                categories_nodes=[],
                categories_segments=[],
                categories_sizes=[],
                default_left=[0] * total,
                id=index,
                left_children=left,
                loss_changes=[0.0] * total,
                parents=parents,
                right_children=right,
                split_conditions=conditions,
                split_indices=split_indices,
                split_type=[0] * total,
                sum_hessian=[1.0] * total,
                tree_param=dict(
                    num_deleted="0",
                    num_feature=str(features),
                    num_nodes=str(total),
                    size_leaf_vector="1",
                ),
            )
        )
    skeleton = xgb.Booster(
        params=dict(objective="reg:squarederror", num_feature=features, base_score=0.125)
    )
    model = json.loads(skeleton.save_raw("json"))
    booster_model = model["learner"]["gradient_booster"]["model"]
    booster_model.update(
        trees=encoded,
        tree_info=[0] * trees,
        iteration_indptr=list(range(trees + 1)),
        gbtree_model_param=dict(num_parallel_tree="1", num_trees=str(trees)),
    )
    booster = xgb.Booster()
    booster.load_model(bytearray(json.dumps(model).encode()))
    booster.set_param(dict(device="cuda:0", nthread=4))
    return booster
