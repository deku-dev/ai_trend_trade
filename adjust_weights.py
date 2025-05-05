import json
import openai
from pathlib import Path

# Ключ API
openai.api_key = "YOUR_API_KEY"

HISTORY_FILE = "history.jsonl"
WEIGHTS_FILE = "weights.json"

def load_history(file, max_entries=20):
    if not Path(file).exists():
        return []
    with open(file, encoding="utf-8") as f:
        return [json.loads(line) for line in f][-max_entries:]

def load_weights(file):
    if not Path(file).exists():
        return {}
    return json.loads(Path(file).read_text(encoding="utf-8"))

def build_prompt(history, current_weights):
    prompt = (
        "Ти виступаєш як система корекції ваг характеристик для передбачення ймовірності трендового руху паперу.\n"
        "Оцінка базується на фічах (features), кожна з яких має свою вагу (weight). Вихід — число від 0 до 100.\n"
        "Враховуючи історичні оцінки GPT та реальні результати, запропонуй нові ваги для фічей, щоб покращити точність.\n\n"
        f"Поточні ваги:\n{json.dumps(current_weights, indent=2, ensure_ascii=False)}\n\n"
        "Ось історія:\n"
    )
    for item in history:
        prompt += f"\nfeatures: {item['features']}\nоцінка: {item['score_given']}\nрезультат: {item['real_outcome']}\n"

    prompt += (
        "\n---\nЗапропонуй нові ваги у форматі JSON та коротко поясни зміни. "
        "Не використовуй нові фічі — тільки ті, що є. Ваги мають сумуватись до 1.0."
    )

    return prompt

def ask_chatgpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response['choices'][0]['message']['content']

def parse_weights_from_response(response):
    try:
        start = response.index('{')
        end = response.rindex('}') + 1
        return json.loads(response[start:end])
    except Exception as e:
        print("Не вдалося розібрати JSON з відповіді GPT.")
        return None

def save_weights(weights, file):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)
    print(f"Нові ваги збережено у {file}")

def main():
    history = load_history(HISTORY_FILE)
    if not history:
        print("Немає історії для аналізу.")
        return

    current_weights = load_weights(WEIGHTS_FILE)
    prompt = build_prompt(history, current_weights)

    print("Надсилаю запит до GPT...\n")
    response = ask_chatgpt(prompt)
    print("GPT відповів:\n", response)

    new_weights = parse_weights_from_response(response)
    if new_weights:
        save_weights(new_weights, WEIGHTS_FILE)

if __name__ == "__main__":
    main()
