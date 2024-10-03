import os
from PIL import Image
from argparse import ArgumentParser
import logging
import id_cat_yolo
import cv2
import shutil


logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('cat finder')


def display_image_cv2(img):
    # Display the image using OpenCV
    cv2.imshow('Object Detection Result', img)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()  # Close the window when a key is pressed



def detect_cat_in_folder(model_module, images_root, num_tags, disp_img):

  ratio = 0.0
  num_images = 0
  num_cat_images = 0
  # Load the MobileNetV2 model pre-trained on ImageNet
  model = model_module.load_model()
  un_id_cat_dir = os.path.join(images_root,'unidentified_cat')
  if not  os.path.isdir(un_id_cat_dir):
     os.mkdir(un_id_cat_dir)
  
  logger.info(f'looking for pics in  {images_root} ')
  for img_name in os.listdir(images_root):
#    logger.debug(f'analyzing {img_name} ')
    if img_name.endswith('.jpg'):      
      img_path = os.path.join(images_root,img_name)
      img, detected_objects = model_module.detect_objects(model, img_path)
      result_image = model_module.draw_boxes(img, detected_objects)
      if disp_img:
        display_image_cv2(result_image)

      num_images += 1
      if (model_module.has_cat(detected_objects)):
        num_cat_images += 1
        logger.info(f'{img_name} has a cat!')
        shutil.move(src=img_path,dst=os.path.join(un_id_cat_dir,img_name))


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

