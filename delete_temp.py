import os

path = './result_roi_8k'

# file_name_list = os.listdir(path)



while 1:
    file_list = sorted(os.listdir(path), key=lambda x: os.path.getmtime(os.path.join(path, x)),reverse=True)
    if len(file_list) > 600:
        for file in file_list[600:]:
            file_path = os.path.join(path,file)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f'删除成功{file_path}')
                except:
                    pass