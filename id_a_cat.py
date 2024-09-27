import os
import tensorflow as tf
from argparse import ArgumentParser
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np



def max_tag_for_cat(image_path, model):

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)


    # Make predictions
    predictions = model.predict(x)
    decoded_predictions = decode_predictions(predictions, top=1000)[0]
    for _, label, score in decoded_predictions:
        if 'cat' in label.lower():
            return True, f"Cat detected with confidence {score:.2f} in image {image_path}"

    return False, f"No cat detected in the image {image_path}"


 

def detect_cat(image_path, model, num_tags):

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)


    # Make predictions
    predictions = model.predict(x)
    decoded_predictions = decode_predictions(predictions, top=num_tags)[0]


    # Check if 'cat' is in the top 3 predictions
    for _, label, score in decoded_predictions:
#        print(f'<== label is {label}')
        if 'cat' in label.lower():
            return True, f"Cat detected with confidence {score:.2f} in image {image_path}"

    return False, f"No cat detected in the image {image_path}"


def detect_cat_in_folder(images_root, num_tags):
  ratio = 0.0
  num_images = 0
  num_cat_images = 0
  # Load the MobileNetV2 model pre-trained on ImageNet
  model = MobileNetV2(weights='imagenet') 
  print(f'looking for pics in  {images_root} ')

  for img_name in os.listdir(images_root):
#    print(f'analyzing {img_name} ')
    if img_name.endswith('.jpg'):      
      img_path = os.path.join(images_root,img_name)
      has_cat, message = detect_cat(img_path, model, num_tags)
      num_images += 1
      if (has_cat):
        num_cat_images += 1
        print(message)

  return float(num_cat_images)/float(num_images)
    



def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--pic_path", 
                        default=os.path.join('c:' ,'temp'), 
                        help="path to where the pictures are stored")

    args = parser.parse_args()
    if os.path.isdir(args.pic_path):
        args.pic_path = os.path.abspath(args.pic_path)
    else:
        raise ValueError('ERROR: Can\'t find ' + args.pic_path)
    return args


args = parse_arguments()

# Example usage
images_root = args.pic_path
cat_res = {}
for cat_root in os.listdir(images_root):

  cat_root = os.path.join(images_root, cat_root)
  if os.path.isdir(cat_root):  
    print(f'cat_root = {cat_root}')    
    cat_res[cat_root] = {}
    num_tags = 0
    ratio = 0.0

    while ratio < 0.99:
      num_tags += 10
      ratio = detect_cat_in_folder(cat_root, num_tags)
      print(f'Ratio is {ratio} for num_tags = {num_tags}')    
      cat_res[cat_root][num_tags] = ratio

for c in cat_res:
   print(f'{c} : {cat_res[c]}')
     
   



 

 

 

 