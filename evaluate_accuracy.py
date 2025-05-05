import json
import openai
from pathlib import Path

# Твій ключ до ChatGPT API
openai.api_key = "YOUR_API_KEY"

HISTORY_FILE = "history.jsonl"

def load_history(file, max_entries=20):
    path = Path(file)
    if not path.exists():
        print("Файл історії не знайдено.")
        return []
    with path.open(encoding="utf-8") as f:
        lines = f.readlines()
        return [json.loads(line) for line in lines][-max_entries:]

def build_prompt(history):
    prompt = (
        "Ти виступаєш як аналітик, який перевіряє ефективність передбачень моделі.\n"
        "У тебе є список прикладів із характеристиками паперу (features), оцінкою GPT, і реальним результатом руху паперу.\n\n"
        "Проаналізуй:\n"
        "- Наскільки точні були оцінки\n"
        "- Де GPT переоцінив або недооцінив\n"
        "- Які фічі вводили GPT в оману\n"
        "- Які зміни можна внести в логіку оцінки або ваги фічей\n\n"
        "Ось приклади:\n"
    )
    for item in history:
        prompt += f"\nfeatures: {item['features']}\nоцінка: {item['score_given']}\nрезультат: {item['real_outcome']}\n"

    prompt += "\nДай висновок українською мовою."

    return prompt

def ask_chatgpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response['choices'][0]['message']['content']

def main():
    history = load_history(HISTORY_FILE)
    if not history:
        return

    prompt = build_prompt(history)
    print("Надсилаю запит до GPT...\n")
    response = ask_chatgpt(prompt)

    print("Висновок GPT:\n")
    print(response)

if __name__ == "__main__":
    main()
