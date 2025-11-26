import pprint

from src.evaluation.models.baselines import PopularityRecommender, SessionPopularityRecommender, RandomRecommender
from src.evaluation.evaluator import evaluate_next_item, evaluate_rest_of_session
from src.evaluation.loader import load_slice_train_test
from src.evaluation.models.itemknn import ItemKNNRecommender
from src.evaluation.models.markov_sr import MarkovRecommender, SequentialRulesRecommender, SecondOrderMarkovRecommender
from src.evaluation.models.sessionknn import SessionKNNRecommender
from src.evaluation.models.vsknn import VSKNNRecommender
from utils import config as cfg
from utils.enviroment import get_path
from utils.paths import PathsEnum


def run_for_model(model_factory, model_name: str, k: int = 10):
    """
    Executa a avaliação de um modelo.

    model_factory: função sem argumentos que retorna uma instância do modelo.
                   Exemplo:
                       lambda: ItemKNNRecommender(k_neighbors=200)
                   Também aceita uma classe, pois classes são "callables":
                       ItemKNNRecommender
    """
    print("=" * 80)
    print(f"Modelo: {model_name}")
    print("=" * 80)

    results_per_slice = []

    for s in range(cfg.N_SLICES):
        print(f"\nSlice {s}")
        train_df, test_df = load_slice_train_test(
            s, split_path=get_path(PathsEnum.MODEL_SPLIT)
        )

        # model_factory() funciona tanto para classes quanto funções
        model = model_factory()

        res = evaluate_next_item(model, train_df, test_df, k=k)

        print(f"  n_events avaliado: {res['n_events']}")
        print(f"  HitRate@{k}: {res[f'hitrate@{k}']:.4f}")
        print(f"  MRR@{k}:     {res[f'mrr@{k}']:.4f}")

        results_per_slice.append({"slice": s, **res})

    avg_hr = sum(r[f"hitrate@{k}"] for r in results_per_slice) / len(results_per_slice)
    avg_mrr = sum(r[f"mrr@{k}"] for r in results_per_slice) / len(results_per_slice)

    print("\nResumo (média sobre slices):")
    print(f"  HitRate@{k}: {avg_hr:.4f}")
    print(f"  MRR@{k}:     {avg_mrr:.4f}")

    print("\nResultados detalhados por slice:")
    pprint.pp(results_per_slice)

def run_for_model_rest(model_factory, model_name: str, k: int = 10):
    print("=" * 80)
    print(f"Modelo (REST-OF-SESSION): {model_name}")
    print("=" * 80)

    results_per_slice = []

    for s in range(cfg.N_SLICES):
        print(f"\nSlice {s}")
        train_df, test_df = load_slice_train_test(
            s, split_path=get_path(PathsEnum.MODEL_SPLIT)
        )

        model = model_factory()
        res = evaluate_rest_of_session(model, train_df, test_df, k=k)

        print(f"  n_cases avaliados: {res['n_cases']}")
        print(f"  Precision@{k}: {res[f'precision@{k}']:.4f}")
        print(f"  Recall@{k}:    {res[f'recall@{k}']:.4f}")
        print(f"  NDCG@{k}:      {res[f'ndcg@{k}']:.4f}")
        print(f"  MAP@{k}:       {res[f'map@{k}']:.4f}")

        results_per_slice.append({"slice": s, **res})

    avg_prec = sum(r[f"precision@{k}"] for r in results_per_slice) / len(results_per_slice)
    avg_rec = sum(r[f"recall@{k}"] for r in results_per_slice) / len(results_per_slice)
    avg_ndcg = sum(r[f"ndcg@{k}"] for r in results_per_slice) / len(results_per_slice)
    avg_map = sum(r[f"map@{k}"] for r in results_per_slice) / len(results_per_slice)

    print("\nResumo (média sobre slices):")
    print(f"  Precision@{k}: {avg_prec:.4f}")
    print(f"  Recall@{k}:    {avg_rec:.4f}")
    print(f"  NDCG@{k}:      {avg_ndcg:.4f}")
    print(f"  MAP@{k}:       {avg_map:.4f}")

def main():
    k = 10
    #
    # # Popularidade global
    # run_for_model_rest(PopularityRecommender, "Popularity", k=k)
    #
    # # Popularidade por sessão
    # run_for_model_rest(SessionPopularityRecommender, "SessionPopularity", k=k)
    #
    # # Random
    # run_for_model_rest(RandomRecommender, "Random", k=k)
    #
    #
    # run_for_model_rest(
    #     lambda: ItemKNNRecommender(k_neighbors=200, max_history=5, use_recency_weight=False),
    #     model_name="ItemKNN",
    #     k=k,
    # )
    #
    # run_for_model_rest(
    #     lambda: SessionKNNRecommender(
    #         n_neighbors=100,
    #         max_history=20,
    #         max_sessions_per_item=1000,
    #         max_candidate_sessions=2000,
    #         use_session_length_normalization=True,
    #         use_recency_weight=False),
    #     "SessionKNN",
    #     k=10,
    # )

    # Markov
    # run_for_model_rest(
    #     lambda: MarkovRecommender(),
    #     "Markov",
    #     k=k,
    # )

    run_for_model_rest(
        lambda: SecondOrderMarkovRecommender(),
        "Markov(2nd)",
        k=k,
    )

    # Sequential Rules
    # run_for_model_rest(
    #     lambda: SequentialRulesRecommender(
    #         max_history=10,
    #         max_steps_ahead=5,
    #         decay=0.7,
    #         min_score=0.0,
    #     ),
    #     "SequentialRules",
    #     k=k,
    # )

    run_for_model_rest(
        lambda: VSKNNRecommender(
            n_neighbors=200,
            max_history=20,
            max_sessions_per_item=1000,
            max_candidate_sessions=2000,
            use_session_length_normalization=True,
            use_recency_weight_session=True,
            use_recency_weight_items=True,
        ),
        "VSKNN",
        k=k,
    )


if __name__ == "__main__":
    main()
