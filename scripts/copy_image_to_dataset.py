import os
import shutil

if __name__ == '__main__':
    start = 5200
    end = 5214
    target_path = os.path.join('.', 'dataset', 'images', 'image')
    IMG_EXTENSIONS = ['.png','.jpg']
    is_image_file = lambda filename : any(filename.endswith(ext) for ext in IMG_EXTENSIONS)
    path = os.path.join('.','raw_data','raw_data_from_005200_to_005699_sort', '005200-005299_sort')
    num_of_file = 0
    for d in sorted(os.listdir(path)):
        if int(d[-6:]) in range(start, end+1):
            for root, _, fnames in sorted(os.walk(os.path.join(path,d))):
                for fn in fnames:
                    if not is_image_file(fn):
                        continue
                    else:
                        if not os.path.exists(os.path.join(target_path, fn)):
                            f = os.path.join(root,fn)
                            num_of_file += 1
                            shutil.copy2(f, target_path)
    print(f"copy {num_of_file} files from {path} to {target_path}")
