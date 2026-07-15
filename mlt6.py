import argparse
import glob
import os

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки (конфигурация одной ветки sweep)."""
    p = argparse.ArgumentParser(description="MLT-06 hyper-parameter sweep branch")
    p.add_argument(
        "--data-dir",
        required=True,
        help="Каталог примонтированного датасета MLT-03 (внутри — папки версий с wine.csv)",
    )
    p.add_argument(
        "--learning-rates",
        default="0.01,0.05,0.1,0.3",
        help="Сетка learning_rate через запятую; индекс ветки выбирает из неё "
        "своё значение (по умолчанию 4 варианта под replicas: 4)",
    )
    p.add_argument("--n-estimators", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def branch_index() -> int:
    """Возвращает индекс ветки sweep (0..N-1).

    Каждая реплика ``replicatedJob`` в JobSet — это отдельный Job, поэтому
    их различает НЕ ``JOB_COMPLETION_INDEX`` (при ``completions: 1`` он у всех
    равен 0), а ``JOB_INDEX``, который JobSet-контроллер инжектит в каждый
    контейнер (0..replicas-1). Оставляем ``JOB_COMPLETION_INDEX`` как фолбэк
    на случай Indexed-Job-варианта запуска.
    """
    raw = os.environ.get("JOB_INDEX") or os.environ.get("JOB_COMPLETION_INDEX") or "0"
    return int(raw)


def load_wine(data_dir: str) -> tuple[pd.DataFrame, pd.Series]:
    """Читает последнюю версию wine.csv и готовит признаки/таргет (quality >= 6)."""
    # Датасет примонтирован целиком: каждая версия — своя папка. Берём последнюю.
    matches = sorted(glob.glob(os.path.join(data_dir, "*", "wine.csv")))
    if not matches:
        # запасной вариант — файл лежит прямо в корне каталога
        matches = sorted(glob.glob(os.path.join(data_dir, "wine.csv")))
    if not matches:
        raise RuntimeError(f"wine.csv не найден под {data_dir}")
    df = pd.read_csv(matches[-1])
    return df.drop(columns=["quality"]), (df["quality"] >= 6)


def main() -> None:
    args = parse_args()

    rates = [float(v) for v in args.learning_rates.split(",") if v.strip()]
    idx = branch_index()
    if idx >= len(rates):
        raise RuntimeError(
            f"branch index {idx} вне сетки learning_rate ({rates}); "
            f"replicas должно совпадать с числом значений"
        )
    lr = rates[idx]
    # Печатаем на входе — сразу видно в логах, что все ветки стартуют совместно
    # (gang) и какая ветка какой learning_rate взяла.
    print(f"JOB_INDEX={idx} learning_rate={lr}", flush=True)

    X, y = load_wine(args.data_dir)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=args.seed)

    # Гиперпараметры конструктора (в т.ч. learning_rate) и метрику training_score
    # фиксирует mlflow.autolog() — своих mlflow-вызовов в коде НЕТ.
    clf = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=lr,
        random_state=args.seed,
    )
    clf.fit(Xtr, ytr)

    print(f"branch {idx} lr={lr} test accuracy: {clf.score(Xte, yte)}", flush=True)


if __name__ == "__main__":
    main()
