import csv
import json
import openai
from pathlib import Path

openai.api_key = "YOUR_API_KEY"

INPUT_FILE = "input.csv"
OUTPUT_FILE = "results.jsonl"
WEIGHTS_FILE = "weights.json"

def load_weights(path):
    if Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return {}

def load_input(path):
    data = []
    with open(path, newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                features = json.loads(row['features'].replace("'", "\""))
                data.append({"ticker": row['ticker'], "features": features})
            except Exception as e:
                print(f"Помилка в рядку {row}: {e}")
    return data

def build_prompt(features, weights):
    prompt = (
        "Ти аналізуєш фінансовий інструмент на основі заданих характеристик (features).\n"
        "Кожна фіча має свою вагу. З урахуванням фічей і ваг, оціни ймовірність трендового руху на шкалі від 0 до 100.\n"
        f"Ваги:\n{json.dumps(weights, ensure_ascii=False)}\n"
        f"Фічі паперу:\n{json.dumps(features, ensure_ascii=False)}\n\n"
        "Відповідь лише у форматі: SCORE: <число від 0 до 100>."
    )
    return prompt

def get_score_from_response(response_text):
    for line in response_text.splitlines():
        if "SCORE:" in line:
            try:
                return float(line.split("SCORE:")[1].strip())
            except:
                continue
    return None

def query_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response['choices'][0]['message']['content']

def save_results(results, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Збережено {len(results)} оцінок у {path}")

def main():
    weights = load_weights(WEIGHTS_FILE)
    papers = load_input(INPUT_FILE)
    results = []

    for paper in papers:
        prompt = build_prompt(paper['features'], weights)
        print(f">>> GPT оцінює {paper['ticker']}...")
        response = query_gpt(prompt)
        score = get_score_from_response(response)
        if score is None:
            print(f"Помилка в відповіді GPT:\n{response}")
            continue
        results.append({
            "ticker": paper["ticker"],
            "features": paper["features"],
            "score_given": score,
            "response": response
        })

    save_results(results, OUTPUT_FILE)

if __name__ == "__main__":
    main()
