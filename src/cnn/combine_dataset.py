import os
import shutil

#rename
i = 0
def rename(path, mode, s):
    for filename in os.listdir(path):
        global i
        dst = f'mat-{s}_({str(i)}).jpg' if mode == 0 else f'mat-{s}_d({str(i)}).jpg'
        src = f'{path}/{filename}'
        dst = f'{path}/{dst}'
        i+=1
        #rename file base on source and destination
        os.rename(src, dst)

path = os.path.join(os.getcwd(), 'src', 'cnn', 'nhat', 'train', 'distract')
rename(path, 1, 'n')
path = os.path.join(os.getcwd(), 'src', 'cnn', 'son', 'train', 'distract')
rename(path, 1, 's')

i=0
path = os.path.join(os.getcwd(), 'src', 'cnn', 'nhat', 'train', 'focus')
rename(path, 0, 'n')
path = os.path.join(os.getcwd(), 'src', 'cnn', 'son', 'train', 'focus')
rename(path, 0, 's')

#merge 2 subfolder to data folder
dist_dst = os.path.join(os.getcwd(), 'src', 'cnn', 'data', 'train', 'distract')
fcs_dst = os.path.join(os.getcwd(), 'src', 'cnn', 'data', 'train', 'focus')

def merge(path0, path1, dst):
    for file in os.listdir(path0):
        current_path = f'{path0}/{file}'
        shutil.copy2(current_path, dst)

    for file in os.listdir(path1):
        current_path = f'{path1}/{file}'
        shutil.copy2(current_path, dst)

nhat_dist = os.path.join(os.getcwd(), 'src', 'cnn', 'nhat', 'train', 'distract')
son_dist = os.path.join(os.getcwd(), 'src', 'cnn', 'son', 'train', 'distract')
merge(nhat_dist, son_dist, dist_dst)

nhat_fcs = os.path.join(os.getcwd(), 'src', 'cnn', 'nhat', 'train', 'focus')
son_fcs = os.path.join(os.getcwd(), 'src', 'cnn', 'son', 'train', 'focus')
merge(nhat_fcs, son_fcs, fcs_dst)
