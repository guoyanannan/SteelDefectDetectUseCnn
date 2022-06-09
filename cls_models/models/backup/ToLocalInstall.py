import os

base_dir = r'/packages'
file_list = os.listdir(base_dir)


def text_save(filename, data):
    file = open(filename, 'a')
    for i in data:
        if i == 'requirements.txt':
            continue
        print(i)
        s = str(i).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")


text_save(r'requirements.txt', file_list)