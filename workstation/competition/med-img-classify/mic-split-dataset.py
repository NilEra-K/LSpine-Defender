import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset_by_class(source_dir, train_dir, valid_dir, test_dir, split_ratio=(0.7, 0.2, 0.1)):
    """
    Splits the dataset by class into training, validation, and test sets while ensuring class distribution is uniform.
    """
    # Create output directories if they don't exist
    for directory in [train_dir, valid_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)

    # Get all class directories
    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    # Initialize counters for each set and class
    set_counters = {train_dir: 0, valid_dir: 0, test_dir: 0}
    class_counters = {train_dir: {}, valid_dir: {}, test_dir: {}}

    # Split data for each class
    for class_dir in class_dirs:
        class_path = os.path.join(source_dir, class_dir)
        class_images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

        # Create a list of labels for stratification (assuming class_dir is the label)
        labels = [class_dir] * len(class_images)

        # Split the images into train and valid_test sets
        train_images, valid_test_images = train_test_split(class_images, test_size=sum(split_ratio[1:]), stratify=labels, random_state=42)

        # Create a new labels list for the valid_test_images
        valid_test_labels = labels[:len(valid_test_images)]

        # Split the valid_test_images into valid and test sets
        valid_images, test_images = train_test_split(valid_test_images, test_size=split_ratio[2]/sum(split_ratio[1:]), stratify=valid_test_labels, random_state=42)

        # 确保每个类的子目录在训练、验证和测试目录中都存在
        for directory in [train_dir, valid_dir, test_dir]:
            class_subdir = os.path.join(directory, class_dir)
            os.makedirs(class_subdir, exist_ok=True)

        # Copy images to their respective directories and update counters
        for images, directory in [(train_images, train_dir),
                                  (valid_images, valid_dir),
                                  (test_images, test_dir)]:
            for image in images:
                src_path = os.path.join(class_path, image)
                dst_path = os.path.join(directory, class_dir, image)
                shutil.copy(src_path, dst_path)
                set_counters[directory] += 1
                if class_dir not in class_counters[directory]:
                    class_counters[directory][class_dir] = 0
                class_counters[directory][class_dir] += 1

    # 输出相关的日志
    for directory, total in set_counters.items():
        print(f"[INFO] Total Images IN {directory}: {total}")
        for class_dir, count in class_counters[directory].items():
            print(f">      {class_dir}: {count}")



# 定义路径
source_dir = 'E:/RSNA-Dataset/mic-images'
train_dir = 'E:/RSNA-Dataset/mic-split/train_images'
valid_dir = 'E:/RSNA-Dataset/mic-split/valid_images'
test_dir = 'E:/RSNA-Dataset/mic-split/test_images'

# 划分数据集
split_dataset_by_class(source_dir, train_dir, valid_dir, test_dir)