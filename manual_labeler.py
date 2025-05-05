import json
from pathlib import Path

RESULTS_FILE = "results.jsonl"
LABELED_FILE = "history.jsonl"

def load_results(file_path):
    if not Path(file_path).exists():
        print("Файл результатів не знайдено.")
        return []
    with open(file_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_labeled(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Збережено {len(data)} записів у {file_path}")

def prompt_real_outcome(ticker, score):
    while True:
        try:
            result = float(input(f"Ticker: {ticker}, GPT-оцінка: {score} — Введи фактичний результат (0–100): "))
            if 0 <= result <= 100:
                return result
            else:
                print("Значення повинно бути від 0 до 100.")
        except ValueError:
            print("Введи число.")

def main():
    data = load_results(RESULTS_FILE)
    if not data:
        return

    labeled = []
    for item in data:
        if "real_outcome" in item:
            labeled.append(item)
            continue
        outcome = prompt_real_outcome(item["ticker"], item["score_given"])
        item["real_outcome"] = outcome
        labeled.append(item)

    save_labeled(labeled, LABELED_FILE)

if __name__ == "__main__":
    main()
