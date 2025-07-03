import os
import shutil
import random

# ПУТИ К ПАПКАМ
images_dir = 'data/raw/images'  # исходные изображения
masks_dir = 'data/raw/masks'  # исходные маски

output_base = 'data'  # если у вас уже есть папки train/val/test

# Соотношение
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Для повторяемости
random.seed(42)

# Список файлов
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Фильтруем только пары
paired_files = []
for img_file in image_files:
    name, _ = os.path.splitext(img_file)
    mask_name = f"{name}_lab.png"
    mask_path = os.path.join(masks_dir, mask_name)

    if os.path.exists(mask_path):
        paired_files.append((img_file, mask_name))
    else:
        print(f"⚠ Маска не найдена для {img_file} -> {mask_name}")

# Перемешаем
random.shuffle(paired_files)

# Делим
n_total = len(paired_files)
n_train = int(train_ratio * n_total)
n_val = int(val_ratio * n_total)

train_files = paired_files[:n_train]
val_files = paired_files[n_train:n_train + n_val]
test_files = paired_files[n_train + n_val:]

# Создать папки
splits = ['train', 'val', 'test']
for split in splits:
    for sub in ['images', 'masks']:
        os.makedirs(os.path.join(output_base, split, sub), exist_ok=True)


# Копировать
def copy_files(file_list, split):
    for img_file, mask_file in file_list:
        shutil.copy(
            os.path.join(images_dir, img_file),
            os.path.join(output_base, split, 'images', img_file)
        )
        shutil.copy(
            os.path.join(masks_dir, mask_file),
            os.path.join(output_base, split, 'masks', mask_file)
        )


copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print("✅ Датасет успешно распределён:")
print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
