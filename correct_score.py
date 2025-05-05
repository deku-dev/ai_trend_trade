import json
from pathlib import Path

RESULTS_FILE = "results.jsonl"
HISTORY_FILE = "history.jsonl"

def read_results():
    if not Path(RESULTS_FILE).exists():
        print("Немає результатів для обробки.")
        return []

    with open(RESULTS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def manual_correction(result):
    print("\n--- Новий запис ---")
    print("Features:", result["features"])
    print("\nGPT відповів:\n", result["gpt_response"])

    while True:
        try:
            corrected_score = int(input("Оціни справжню ймовірність тренду (0-100): "))
            if 0 <= corrected_score <= 100:
                break
        except ValueError:
            pass
        print("Введи число від 0 до 100.")

    outcome = input("Результат: (успіх/провал/інше): ").strip().lower()
    if not outcome:
        outcome = "unknown"

    return {
        "features": result["features"],
        "score_given": corrected_score,
        "real_outcome": outcome
    }

def append_to_history(entry):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def main():
    results = read_results()
    if not results:
        return

    for res in results:
        entry = manual_correction(res)
        append_to_history(entry)

    print("\nУспішно оновлено історію!")

if __name__ == "__main__":
    main()
