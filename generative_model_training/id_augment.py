import os
import shutil


def moveFiles(path, disdir, suffix = ".png"):  # path为原始路径，disdir是移动的目标目录
    for j in range(0,5):
        imagename = os.path.join(disdir, str(j + 50) + suffix)  
        shutil.copy(path, imagename)

id_images_root = "../dataset/context_database/images"
disdir = './samples'    
suffix = ".png"
# load id images
if os.path.isdir(id_images_root):
    id_images = os.listdir(id_images_root)
    id_images = sorted(id_images)
    ## add origin id x 5 ###
    for i in range(0, len(id_images)):
        origin_id = id_images[i]
        disdir_id = os.path.join(disdir, str(i))
        if os.path.isdir(disdir_id):
            moveFiles(os.path.join(id_images_root, origin_id), disdir_id, suffix = suffix)
        if i % 100 == 0:
            print("done %d"%i)