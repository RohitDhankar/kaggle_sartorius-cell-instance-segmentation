
## SOURCE -- Mostly from - KAGGLE 
## KAGGLE Code URL -- https://www.kaggle.com/datasets/slawekbiel/sartorius-cell-instance-segmentation-coco

from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image as PIL_Image
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import json, itertools
import cv2


from datetime import datetime

dt_time_now = datetime.now()
dt_time_save = dt_time_now.strftime("_%Y_%m_%d__%H_%M_")


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


class get_rle():
    """

    # ORIGINAL SOURCE --  https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
    # Original COCO Code File included in this REPO 

    RLE is a simple yet efficient format for storing binary masks. 
    RLE first divides a vector (or vectorized image) into a series of piecewise constant regions 
    and then for each piece simply stores the length of that piece. 
    For example, 
    given M=[0 0 1 1 1 0 1] the RLE counts would be [2 3 1 1], 
    or for M=[1 1 1 1 1 1 0] the RLE counts would be [0 6 1] 
    (note that the odd counts are always the numbers of zeros).

    Returns:
        : 
    
    """

    def __init__(self):
        pass

    def rle_decode(self , mask_rle, shape):
        '''
        #SOURCE -  https://www.kaggle.com/stainsby/fast-tested-rle

        Args:
            mask_rle:  The values from- each ROW for the COLUMN Named -- annotation , from the train.csv 
            mask_rle [Original Definition] :  run-length as string formated (start length)
            shape:     (height,width) of array to return 
        
        Returns: 
            numpy array:    1 - mask, 0 - background

        '''
        print("  "*50)
        print("--mask_rle--",mask_rle)
        ls_split_mask_rle = mask_rle.split()
        
        print("--split_mask_rle--\n",ls_split_mask_rle)
        ls_starts, ls_lengths = [np.asarray(x, dtype=int) for x in (ls_split_mask_rle[0:][::2], ls_split_mask_rle[1:][::2])]
        print("--mask_rle-->  LENGTH -- ls_starts, ls_lengths",len(ls_starts), len(ls_lengths))
        # TODO -- these lengths VARY between - 10 to 59 
        print("--mask_rle--> ls_starts, ls_lengths",ls_starts, ls_lengths)

        ls_starts -= 1
        ls_ends = ls_starts + ls_lengths
        print("--mask_rle--> ls_ends",ls_ends)
        print("--mask_rle-->  LENGTH -- ls_ends",len(ls_ends))

        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        cv2.imwrite("return_img_rle_aa.png",img)

        for lo, hi in zip(ls_starts, ls_ends):
            img[lo:hi] = 1
        return_img_rle = img.reshape(shape)
        cv2.imwrite("return_img_rle_.png",return_img_rle)
        print("--return_img_rle.ndim,return_img_rle.shape----\n",return_img_rle.ndim,return_img_rle.shape)
        print("--return_img_rle---\n",return_img_rle)
        print("  "*50)
        print(" -- "*50)


        return return_img_rle # Needed to align to RLE direction

        
    def binary_mask_to_rle(self , binary_mask):
        '''
        #SOURCE -  https://newbedev.com/encode-numpy-array-using-uncompressed-rle-for-coco-dataset

        Args:
            mask_rle:  run-length as string formated (start length)
            shape:     (height,width) of array to return 
        
        Returns: 
            numpy array:    1 - mask, 0 - background

        '''
        rle = {'counts': [], 'size': list(binary_mask.shape)}
        counts = rle.get('counts')
        for i, (value, elements) in enumerate(itertools.groupby(binary_mask.ravel(order='F'))):
            if i == 0 and value == 1:
                counts.append(0)
            counts.append(len(list(elements)))
        return rle

    def coco_structure(self , train_df):
        """
        Args:

        Returns:
                {'categories':cats, 'images':images,'annotations':annotations}
        """

        obj_init_pycoco = init_pycoco_tools()
        train_dir_path = obj_init_pycoco.img_data_dir_train
        print("--coco_structure-----train_dir_path---\n",train_dir_path)
        lower_white = np.uint8([5, 5, 5]) # TODO -- best range
        upper_white = np.uint8([255,11,255]) # TODO -- best range
        


        cat_ids = {name:id+1 for id, name in enumerate(train_df.cell_type.unique())}    
        cats =[{'name':name, 'id':id} for name,id in cat_ids.items()] 
        #KEY_NAME == categories | KEY_VALUE == LIST == cats
        images = [{'id':id, 'width':row.width, 'height':row.height, 'file_name':f'train/{id}.png'} for id,row in train_df.groupby('id').agg('first').iterrows()]
        ##KEY_NAME == images | KEY_VALUE == LIST == images
        annotations=[]
        for idx, row in tqdm(train_df.iterrows()):
            mk = self.rle_decode(row.annotation, (row.height, row.width))
            print("  "*50)
            print(" -- "*50)

            print("--return_img_rle-----mk---\n",mk)
            unique, counts = np.unique(mk, return_counts=True)
            print("  "*50)
            print("--return_img_rle-----mk---unique, counts--\n",unique, counts)
            print("  "*50)
            print(" -- "*50)
            # cv2.imshow("Left", mk) # TODO - Ok with - PY 3.6 ? 
            # cv2.waitKey(0)
            for image_id,df_csv_row in train_df.groupby('id').agg('first').iterrows():
                print("----IMAGE--ID---",image_id)
            train_img_file_path = str(train_dir_path) + "/" + "train/"+str(image_id)+".png"
            image_input = io.imread(train_img_file_path)
            print("---image_input---CHANNELS---",image_input.shape)
            #img_hsv = cv2.cvtColor(image_input,cv2.COLOR_BGR2HSV)
            #img_gray = cv2.cvtColor(image_input,cv2.COLOR_BGR2GRAY)
            #mask_white = cv2.inRange(image_input, lower_white, upper_white) ##

            masked_image = cv2.bitwise_and(image_input,image_input,mask = mk) #mask_white)
            masked_image[mk==0] = [100]#, 255]#, 255] 
            masked_image[mk==1]= [20] #,:] = [0, 0]#, 1]
            cv2.imwrite("test_masked_image_"+str(counts[1])+".png",masked_image)

  
            ys, xs = np.where(mk)
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            enc = self.binary_mask_to_rle(mk)
            seg = {
                'segmentation':enc, 
                'bbox': [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)],
                'area': int(np.sum(mk)),
                'image_id':row.id, 
                'category_id':cat_ids[row.cell_type], 
                'iscrowd':0, 
                'id':idx
            }
            annotations.append(seg)
        root_dict = {'categories':cats, 'images':images,'annotations':annotations}
        #print("--INFO----TYPE(root_dict-----",type(root_dict)) # DICT 
        
        for key, val in root_dict.items():
            print("---root_dict--KEY, root_dict-val--TYPE---\n",key,type(val))


        return root_dict #{'categories':cats, 'images':images,'annotations':annotations}

    def read_train_csv(self,train_df_path,read_rows_count):
        """ 
        # run on first three images for demonstration
        """
        ls_of_DF_chunks = []
        
        train_df = pd.read_csv(train_df_path)
        print("--[INFO--read_train_csv]---Rows Count--train_df---",train_df.shape[0])
        for csv_chunk in pd.read_csv(train_df_path,chunksize=read_rows_count,iterator=True, low_memory=False):
            ls_of_DF_chunks.append(csv_chunk)
            #print("-[INFO--read_train_csv]---csv_chunk.info----",csv_chunk.info()) ## DONT
            #break -- BREAK out of LOOP -- if getting only 1 CHUNK == csv_chunk
        chunk_idx = 2 
        train_df = ls_of_DF_chunks[chunk_idx]
        print("-[INFO--read_train_csv]---len(ls_of_DF_chunks)--",len(ls_of_DF_chunks))
        print("-[INFO--read_train_csv]---train_df.info----",train_df.info())
        # Write a small sample CSV to see manually 
        train_df.to_csv("./output_dir/csv_dir/"+"df_small_train_"+str(dt_time_save)+"_.csv",index=False)

        #

        all_ids = train_df.id.unique()
        print("-[INFO--read_train_csv]---all_ids.info----",type(all_ids))

        train_sample = train_df[train_df.id.isin(all_ids[:3])]
        root = self.coco_structure(train_sample) #root_dict
        print("--TYPE-root--",type(root))
        print("---root--\n",root)

        with open('./output_dir/json_dir/annos_sample'+str(dt_time_save)+'_.json', 'w', encoding='utf-8') as f:
            json.dump(root, f, ensure_ascii=True, indent=4)


if __name__ == "__main__":
    # TODO -- CHECK -- https://github.com/okotaku/dethub/blob/main/tools/dataset_converters/prepare_sartorius_cellseg.py
    train_df_path = "./input_dir/img_dir/train.csv"
    read_rows_count = 50
    # obj_init_pycoco = init_pycoco_tools()
    # obj_init_pycoco._load_display()
    obj_get_rle = get_rle()
    obj_get_rle.read_train_csv(train_df_path,read_rows_count)


