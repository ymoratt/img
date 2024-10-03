import os
from PIL import Image
from argparse import ArgumentParser
import logging
import id_cat_yolo
import cv2
import shutil


import tkinter as tk
from tkinter import messagebox



logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('cat finder')
#global selected_cat_label

def display_image_cv2(img):
    # Display the image using OpenCV
    cv2.imshow('Object Detection Result', img)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()  # Close the window when a key is pressed


def get_cat_listbox(root):
#  root = tk.Tk()
  Cat_names = ('Izevel', 'Soda', 'Pandi', 'Mark', 'Evil', 'Dubi', 'Gingi', 'Other')
  cat_var = tk.Variable(value=Cat_names)
  cat_listbox = tk.Listbox(
    root,
    listvariable=cat_var,
    height=6,
    selectmode=tk.EXTENDED )
  
  cat_listbox.pack(expand=True, fill=tk.BOTH)
  def items_selected(event):
    root.quit()

  cat_listbox.bind('<<ListboxSelect>>', items_selected)
  return cat_listbox

   

def get_label_from_user(cat_listbox):
  
  root = tk.Tk()
  cat_listbox= get_cat_listbox(root)
  
  root.mainloop()
  selected_indices = cat_listbox.curselection()
  selected_cat_label = ",".join([cat_listbox.get(i) for i in selected_indices])
  logger.info(f'After main loop: {selected_cat_label}')
  cat_listbox.selection_clear(0, tk.END)  # Clear the selection
  root.destroy()
  return selected_cat_label


def detect_cat_in_folder(model_module, images_root, num_tags, disp_img):

  ratio = 0.0
  num_images = 0
  num_cat_images = 0
  # Load the MobileNetV2 model pre-trained on ImageNet
  model = model_module.load_model()
  un_id_cat_dir = os.path.join(images_root,'unidentified_cat')
  if not  os.path.isdir(un_id_cat_dir):
     os.mkdir(un_id_cat_dir)
  no_cat_dir = os.path.join(images_root,'no_cat')
  if not  os.path.isdir(no_cat_dir):
     os.mkdir(no_cat_dir)

  # cat_listbox = get_cat_listbox()
  cat_listbox = None

  logger.info(f'looking for pics in  {images_root} ')
  for img_name in os.listdir(images_root):
#    logger.debug(f'analyzing {img_name} ')
    if img_name.endswith('.jpg'):
      num_images += 1
      img_path = os.path.join(images_root,img_name)
      img, detected_objects = model_module.detect_objects(model, img_path)
      if (model_module.has_cat(detected_objects)):
        num_cat_images += 1
        if disp_img:
          result_image = model_module.draw_boxes(img, detected_objects)
          display_image_cv2(result_image)

          image_label = get_label_from_user(cat_listbox)
#          model_module.label_image(img_path, img, detected_objects, image_label)
          logger.info(f'{img_name} has a cat named {image_label}!')
#TODO        shutil.move(src=img_path,dst=os.path.join(un_id_cat_dir,img_name))
      else:  
        # No Cat
        shutil.move(src=img_path,dst=os.path.join(no_cat_dir,img_name))



  return float(num_cat_images)/float(num_images)


def load_model_module(model):
  if model == 'YOLO':
    return id_cat_yolo
  
  return None
      
   


def parse_arguments():
    supported_models = ['YOLO']
    parser = ArgumentParser()
    parser.add_argument("--pic_path", 
                        default=os.path.join('c:\\' ,'temp'), 
                        help="path to where the pictures are stored")
    parser.add_argument("--model", 
                        default='YOLO', 
                        choices=supported_models,
                        help="Which model to use")

    parser.add_argument("--display_images", 
                        default=False, 
                        action='store_true',
                        help="set display_images to display each image with its detected objects")


    args = parser.parse_args()
    if os.path.isdir(args.pic_path):
        args.pic_path = os.path.abspath(args.pic_path)
    else:
        raise ValueError('ERROR: Can\'t find ' + args.pic_path)
    return args



if __name__ == '__main__':
  # Configure the logging system
  args = parse_arguments()
  images_root = args.pic_path
  model_module = load_model_module(args.model)
  detect_cat_in_folder(model_module=model_module, images_root=images_root, num_tags=3, disp_img=args.display_images)

