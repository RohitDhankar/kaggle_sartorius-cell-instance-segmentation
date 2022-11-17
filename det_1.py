
## SOURCE -- Mostly from - KAGGLE 
## KAGGLE Code URL -- https://www.kaggle.com/datasets/slawekbiel/sartorius-cell-instance-segmentation-coco

from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image as PIL_Image

class init_pycoco_tools():
    def __init__(self):
        self.img_data_dir_test=Path('./input_dir/img_dir/test/')
        self.img_data_dir_train=Path('./input_dir/img_dir/')#train/')
        anno_json_File = Path('./input_dir/json_dir/annotations_val.json')
        self.coco_obj = COCO(anno_json_File)
        imgIds = self.coco_obj.getImgIds()
        self.imgs = self.coco_obj.loadImgs(imgIds[-3:])
        #input_dir/img_dir/test/7ae19de7bc2a.png
        
    def _load_display(self):
        """  """    

        try:
            imgs = self.imgs
            print("---LEN IMAGES --- ",len(imgs))
            coco_obj = self.coco_obj

            _,axs = plt.subplots(len(imgs),2,figsize=(40,15 * len(imgs)))
            print("---LEN axs --- ",axs)

            for img, ax in zip(imgs, axs):
                
                img_file_path = img['file_name']
                print("--INFO--img_file_path--from JSON ANNOTATIONS FILE---",img_file_path)
                img_file_name = img_file_path.rsplit("train/",1)[1]
                img_file_name = img_file_name.rsplit(".png",1)[0]

                print("--INFO--img_file_name--from JSON ANNOTATIONS FILE---",img_file_name)

                #if len(img_file_name) > 0:
                train_img_file_path = str(self.img_data_dir_train) + "/" + str(img_file_path)

                I = io.imread(train_img_file_path)
                print("---TYPE--I---",type(I))

                annIds = coco_obj.getAnnIds(imgIds=[img['id']])
                print("--annIds---",type(annIds))
                print("--annIds--aaa-",annIds)
                #
                anns = coco_obj.loadAnns(annIds)
                print("--ANNO---",type(anns))
                print("--ANNO---",len(anns))

                ax[0].imshow(I)
                print("---type---",type(ax[0]))
                ax[1].imshow(I)
                im_1 = PIL_Image.fromarray(I)
                im_1.save("test_img_"+str(img_file_name)+"_.png")
                plt.sca(ax[1])
                coco_obj.showAnns(anns, draw_bbox=True)
            plt.savefig('test_anno_plots_'+str(img_file_name)+'_.png', bbox_inches='tight', pad_inches=0)

        except Exception as err_load_display:
            print('--[ERROR]--err_load_display--\n',err_load_display)
            pass


if __name__ == "__main__":
    obj_init_pycoco = init_pycoco_tools()
    obj_init_pycoco._load_display()

