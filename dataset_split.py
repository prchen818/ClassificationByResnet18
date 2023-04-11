import glob
import os.path
import random
from PIL import Image

if __name__ =='__main__':
    test_split_ratio = 0.05
    desired_size = 128
    src_path = './dataset'

    dirs = glob.glob(os.path.join(src_path, '*'))
    dirs = [d for d in dirs if os.path.isdir(d)]

    print(f'totally {len(dirs)} classes:\n {dirs}')

    for path in dirs:
        path = path.split('/')[-1]

        os.makedirs(f"train/{path}", exist_ok=True)
        os.makedirs(f"test/{path}", exist_ok=True)

        files = glob.glob(os.path.join(src_path, path, '*.jpg'))
        files += glob.glob(os.path.join(src_path, path, '*.JPG'))
        files += glob.glob(os.path.join(src_path, path, '*.png'))

        # 如果要重新执行数据集划分，应该把原有的训练集和测试集删除
        # 因为每次划分是都随机的，导致每次被划到测试集的数据并不相同
        random.shuffle(files) # 随机打乱顺序

        boundary = int(len(files)*test_split_ratio)
        print("")
        for i, file in enumerate(files):
            img = Image.open(file).convert('RGB')

            old_size=img.size # (width, heigth)
            ratio = float(desired_size)/max(old_size)
            new_size = (int(old_size[0]*ratio), int(old_size[1]*ratio))

            im = img.resize(new_size, Image.LANCZOS)
            new_im = Image.new("RGB", (desired_size, desired_size))
            new_im.paste(im, ((desired_size-new_size[0])//2,
                              (desired_size-new_size[1])//2))

            assert new_im.mode == 'RGB'

            if i <= boundary:
                new_im.save(os.path.join(f'test/{path}', file.split('/')[-1].split('.')[0]+'.jpg'))
            else:
                new_im.save(os.path.join(f'train/{path}', file.split('/')[-1].split('.')[0] + '.jpg'))

            print(f"\rconverting {path} images: {i + 1}/{len(files)}", end='', flush=True)
