from pathlib import Path
import random

import numpy as np
import pandas as pd

from deap import algorithms, base, creator, tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import (
    load_data,
    prepare_features,
    build_preprocessor,
)


# -------------------------------
# 1. Configuration
# -------------------------------
RANDOM_SEED = 42
POPULATION_SIZE = 20
GENERATIONS = 5
CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.2

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# -------------------------------
# 2. Load and split training data
# -------------------------------
train_df, _ = load_data()

X, y = prepare_features(train_df)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y,
)

feature_names = X.columns.tolist()
number_of_features = len(feature_names)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Original input features:", number_of_features)


# -------------------------------
# 3. Define fitness function
# -------------------------------
def evaluate_individual(individual):
    selected_indices = [
        i for i, bit in enumerate(individual) if bit == 1
    ]

    # Reject an empty feature subset
    if not selected_indices:
        return (0.0,)

    selected_features = [
        feature_names[i] for i in selected_indices
    ]

    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]

    model = Pipeline([
        (
            "preprocessor",
            build_preprocessor(X_train_selected),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=50,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                class_weight="balanced",
            ),
        ),
    ])

    model.fit(X_train_selected, y_train)
    predictions = model.predict(X_val_selected)

    validation_f1 = f1_score(
        y_val,
        predictions,
        zero_division=0,
    )

    # Reward good F1 and penalize large feature subsets
    feature_ratio = len(selected_indices) / number_of_features
    fitness = validation_f1 - 0.05 * feature_ratio

    return (fitness,)


# -------------------------------
# 4. Set up Genetic Algorithm
# -------------------------------
if not hasattr(creator, "FitnessMaxNIDS"):
    creator.create(
        "FitnessMaxNIDS",
        base.Fitness,
        weights=(1.0,),
    )

if not hasattr(creator, "IndividualNIDS"):
    creator.create(
        "IndividualNIDS",
        list,
        fitness=creator.FitnessMaxNIDS,
    )

toolbox = base.Toolbox()

toolbox.register("bit", random.randint, 0, 1)

toolbox.register(
    "individual",
    tools.initRepeat,
    creator.IndividualNIDS,
    toolbox.bit,
    n=number_of_features,
)

toolbox.register(
    "population",
    tools.initRepeat,
    list,
    toolbox.individual,
)

toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register(
    "mutate",
    tools.mutFlipBit,
    indpb=1.0 / number_of_features,
)
toolbox.register("select", tools.selTournament, tournsize=3)


# -------------------------------
# 5. Run the GA
# -------------------------------
population = toolbox.population(n=POPULATION_SIZE)

hall_of_fame = tools.HallOfFame(1)

statistics = tools.Statistics(
    lambda individual: individual.fitness.values[0]
)
statistics.register("avg", np.mean)
statistics.register("max", np.max)

print("\nStarting Genetic Algorithm search...")

population, logbook = algorithms.eaSimple(
    population,
    toolbox,
    cxpb=CROSSOVER_PROBABILITY,
    mutpb=MUTATION_PROBABILITY,
    ngen=GENERATIONS,
    stats=statistics,
    halloffame=hall_of_fame,
    verbose=True,
)


# -------------------------------
# 6. Save selected features
# -------------------------------
best_individual = hall_of_fame[0]

selected_features = [
    feature_names[i]
    for i, bit in enumerate(best_individual)
    if bit == 1
]

Path("results").mkdir(exist_ok=True)

pd.Series(
    selected_features,
    name="selected_feature",
).to_csv(
    "results/ga_selected_features.csv",
    index=False,
)

print("\n===== GA FEATURE SELECTION RESULTS =====")
print("Original feature count:", number_of_features)
print("Selected feature count:", len(selected_features))
print("Selected features:", selected_features)
print("Best fitness:", best_individual.fitness.values[0])
print("Saved features to results/ga_selected_features.csv")
