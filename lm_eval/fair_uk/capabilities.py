"""Task requirements and language selection, independent of model names."""

from lm_eval.fair_uk.data import PROTOCOLS, family


BENCHMARKS = {
    "warbias": {"uk": "warbias_uk", "en": "warbias_en"},
    "warbias_intersectional": {
        "uk": "warbias_intersectional_uk",
        "en": "warbias_intersectional_en",
    },
    "bbq": {"uk": "bbq_uk", "en": "bbq_en"},
    "stereoset": {"uk": "stereoset_uk", "en": "stereoset_en"},
    "winobias_natural": {"uk": "winobias_uk_natural", "en": "winobias_en_natural"},
    "winobias_controlled": {
        "uk": "winobias_uk_controlled",
        "en": "winobias_en_controlled",
    },
}
DEFAULT_BENCHMARKS = tuple(BENCHMARKS)
BENCHMARKS.update(
    {
        name: {language: f"{name}_{language}" for language in ("uk", "en")}
        for name in (
            "warbias_triplets",
            "warbias_cross_actor",
            "warbias_expanded_qa",
            "warbias_benign",
        )
    }
)


def benchmark(task):
    for name, tracks in BENCHMARKS.items():
        if task in tracks.values():
            return name
    raise ValueError(f"Unknown benchmark task: {task}")


def task_metadata(task):
    kind = family(task)
    return {
        "benchmark": benchmark(task),
        "protocol": PROTOCOLS[kind],
        "required_capability": "generate_text"
        if kind in ("warbias", "warbias_benign")
        else "score_tokens",
        "requests_per_row": {
            "warbias": 1,
            "warbias_triplets": 3,
            "warbias_benign": 1,
            "bbq_uk": 9,
            "stereoset_uk": 3,
            "winobias_uk_natural": 2,
            "winobias_uk_controlled": 2,
        }[kind],
        "counterpart": next(
            t for t in BENCHMARKS[benchmark(task)].values() if t != task
        ),
    }


def resolve_tasks(registry, tasks=None, languages=None, benchmarks=None):
    if tasks and benchmarks:
        raise ValueError("Select --tasks or --benchmarks, not both")
    if tasks and len(tasks) != len(set(tasks)):
        raise ValueError("Select each task only once")
    if languages and len(languages) != len(set(languages)):
        raise ValueError("Select each language only once")
    if tasks and not languages:
        selected = list(tasks)
    else:
        names = benchmarks or (
            list(dict.fromkeys(benchmark(t) for t in tasks))
            if tasks
            else list(DEFAULT_BENCHMARKS)
        )
        if len(names) != len(set(names)):
            raise ValueError("Select each benchmark only once")
        selected = [
            BENCHMARKS[name][lang] for name in names for lang in (languages or ["uk"])
        ]
    missing = [task for task in selected if task not in registry]
    return selected, missing


def validate_backend(tasks, backend):
    if backend not in ("hf", "vllm", "api"):
        raise ValueError("Choose a supported backend interface: hf, vllm or api")
    unsupported = [
        task
        for task in tasks
        if backend == "api"
        and task_metadata(task)["required_capability"] != "generate_text"
    ]
    if unsupported:
        raise ValueError(
            "This hosted API adapter generates text and cannot score supplied candidate tokens: "
            + ", ".join(unsupported)
        )
