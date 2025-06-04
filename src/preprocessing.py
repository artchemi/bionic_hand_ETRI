import requests
from tqdm import tqdm
import zipfile
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))    # Импортируем корневую директорию


# def main():
#     url = "https://ninapro.hevs.ch/files/DB5_Preproc/s1.zip"
#     filename = "s1.zip"

#     response = requests.get(url, stream=True)
#     total_size = int(response.headers.get('content-length', 0))
#     block_size = 1024  # 1 КБ

#     if response.status_code == 200:
#         with open(filename, 'wb') as f, tqdm(
#             total=total_size,
#             unit='iB',
#             unit_scale=True,
#             desc=filename
#         ) as bar:
#             for chunk in response.iter_content(chunk_size=block_size):
#                 if chunk:  # фильтрация keep-alive
#                     f.write(chunk)
#                     bar.update(len(chunk))
#         print("✅ Скачивание завершено.")
#     else:
#         print(f"❌ Ошибка при загрузке: {response.status_code}")


def main():
    for i in range(1, 11):
        zip_path = f'data/s{i}.zip'
        extract_to = f'data/s{i}'

        os.makedirs(extract_to, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        print(f"{zip_path} --- unpacked")

if __name__ == "__main__":
    main()