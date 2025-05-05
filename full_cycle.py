import subprocess
import time
from pathlib import Path

def run(script_name):
    print(f"\n>>> Виконується {script_name}...")
    subprocess.run(["python", script_name])

def wait_for_user_confirmation():
    print("\n>>> Завершено генерацію оцінок. Проведи ручну перевірку результатів у `results.jsonl`.")
    input("Після корекції вручну натисни Enter, щоб продовжити...")

def check_results_file():
    if not Path("results.jsonl").exists():
        print("Файл results.jsonl відсутній. Спочатку запусти генерацію оцінок.")
        return False
    return True

def main():
    run("run.py")  # Генерація оцінок GPT
    wait_for_user_confirmation()  # Очікує ручну корекцію
    if not check_results_file():
        return
    run("correct_score.py")       # Збір фактичних оцінок
    run("evaluate_accuracy.py")   # GPT аналізує помилки
    run("adjust_weights.py")      # GPT оновлює ваги

    print("\n>>> Цикл завершено. Ваги оновлено, GPT навчився на нових даних.")

if __name__ == "__main__":
    main()
