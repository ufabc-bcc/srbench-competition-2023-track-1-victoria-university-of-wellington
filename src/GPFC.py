import numpy as np
import pandas as pd
from sympy import simplify, latex, count_ops

from evolutionary_forest.forest import EvolutionaryForestRegressor
from evolutionary_forest.utils import reset_random

gpfc = EvolutionaryForestRegressor(
    n_gen=20, n_pop=200, select='AutomaticLexicase',
    cross_pb=0.9, mutation_pb=0.1, max_height=3,
    boost_size=1, initial_tree_size='0-2', gene_num=1,
    mutation_scheme='EDA-Terminal-PM',
    basic_primitives=','.join(['Add', 'Sub', 'Mul', 'Div']),
    base_learner='RidgeCV', verbose=False, normalize=False,
    external_archive=1, root_crossover=True
)

reset_random(0)
task_ids = [1, 2, 3]

for task_id in task_ids:
    # Read the dataset
    data = pd.read_csv(f'datasets/dataset_{task_id}.csv')
    X = np.array(data.drop(['y'], axis=1))
    y = np.array(data['y'])
    if task_id == 3:
        print(np.mean(y))
    else:
        gpfc.fit(X, y)
        simplified_model = simplify(gpfc.model())
        print('Node Count', count_ops(simplified_model))
        print(simplified_model)
        latex_string = latex(simplified_model)
        print(latex_string)
