import zipfile
import tarfile
import rarfile
import py7zr

import shutil
import os

def remove_extension(file_path):
    while os.path.splitext(file_path)[1]:  # 检查是否有扩展名
        file_path = os.path.splitext(file_path)[0]
    return file_path


def extract_file(file_path, extract_to):
    """
    自动识别并解压文件
    :param file_path: 压缩文件的路径
    :param extract_to: 解压目标目录
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    try:
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                print(f">>> [LOGS] zip 文件已解压到 {extract_to}")
        elif file_path.endswith(('.tar', '.tar.gz', '.tgz')):
            with tarfile.open(file_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
                print(f">>> [LOGS] tar 文件已解压到 {extract_to}")
        elif file_path.endswith('.rar'):
            with rarfile.RarFile(file_path, 'r') as rar_ref:
                rar_ref.extractall(extract_to)
                print(f">>> [LOGS] rar 文件已解压到 {extract_to}")
        elif file_path.endswith('.7z'):
            with py7zr.SevenZipFile(file_path, 'r') as sevenz_ref:
                sevenz_ref.extractall(extract_to)
                print(f">>> [LOGS] 7z 文件已解压到 {extract_to}")
        else:
            print("不支持的文件格式")
    except zipfile.BadZipFile:
        print(">>> [ERROR] zip 文件损坏")
    except tarfile.TarError:
        print(">>> [ERROR] tar 文件损坏")
    except rarfile.Error:
        print(">>> [ERROR] rar 文件损坏或需要密码")
    except py7zr.Bad7zFile:
        print(">>> [ERROR] 7z 文件损坏或需要密码")
    except Exception as e:
        print(f">>> [ERROR] 解压失败: {e}")


def delete_folder(folder_path):
    """
    删除指定文件夹及其所有内容（包括子文件夹和文件）。
    :param folder_path: 要删除的文件夹路径
    """
    try:
        # 检查路径是否存在
        if os.path.exists(folder_path):
            # 删除文件夹及其所有内容
            shutil.rmtree(folder_path)
            print(f">>> [IMPORTANT] 文件夹 {folder_path} 及其所有内容已被删除。")
        else:
            print(f">>> [ERROR] 路径 {folder_path} 不存在，无需删除。")
    except Exception as e:
        print(f">>> [ERROR] 删除文件夹时发生错误: {e}")


def delete_file(file_path):
    """
    直接删除指定文件，无需用户确认。
    :param file_path: 要删除的文件路径
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f">>> [IMPORTANT] 文件 {file_path} 已成功删除。")
        else:
            print(f">>> [ERROR] 文件 {file_path} 不存在，无需删除。")
    except Exception as e:
        print(f">>> [ERROR] 删除文件时发生错误: {e}")
        
def delete_files(file_paths):
    """
    批量删除多个文件，无需用户确认。
    :param file_paths: 要删除的文件路径列表
    """
    for file_path in file_paths:
        delete_file(file_path)



if __name__ == "__main__":
    # 使用示例
    file_path = "example.zip"
    extract_to = "extracted_folder"         # 解压到的目标文件夹
    extract_file(file_path, extract_to)
