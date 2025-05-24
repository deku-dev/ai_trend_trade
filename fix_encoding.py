import os
import json

OUTPUT_DIR = 'output/'

def fix_json_files():
    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith('.json'):
            file_path = os.path.join(OUTPUT_DIR, filename)
            print(f'Виправлення файлу: {file_path}')
            try:
                # Відкриваємо файл у байтовому режимі
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                # Спроба декодувати в UTF-8
                try:
                    content = content.decode('utf-8')
                except UnicodeDecodeError:
                    print(f"Не вдалось декодувати UTF-8. Пробую cp1252...")
                    content = content.decode('cp1252')
                
                # Завантажуємо як JSON
                data = json.loads(content)
                
                # Зберігаємо з параметром ensure_ascii=False
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                
                print(f'Файл {filename} успішно виправлений.')
            except Exception as e:
                print(f'Помилка при обробці файлу {filename}: {e}')

if __name__ == '__main__':
    fix_json_files()
