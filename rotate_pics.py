import os
from PIL import Image
from argparse import ArgumentParser


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




def rotate_images(directory):
    # Iterate through all files in the given directory
    for filename in os.listdir(directory):
        # Check if the file is a JPG image
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            
            try:
                # Open the image
                with Image.open(file_path) as img:
                    # Rotate the image 180 degrees
                    rotated_img = img.rotate(180)
                    
                    # Save the rotated image, overwriting the original
                    rotated_img.save(file_path)
                    
                print(f"Successfully rotated and saved: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    args = parse_arguments()
    rotate_images(args.pic_path)
        