import os
import paramiko
import time
import cv2
import datetime

def open_connection(server, username, password):
  ssh = paramiko.SSHClient() 
  ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
  ssh.connect(server, username=username, password=password)
  sftp = ssh.open_sftp()
  return ssh, sftp

def close_connection(ssh, sftp):
  sftp.close()  
  ssh.close()
  

def get_image(sftp, localpath, remotepath):
  # sftp.put(localpath, remotepath)
  sftp.get(localpath=localpath, remotepath=remotepath)

def del_remote_image(ssh_client, remotepath):
  assert(remotepath.endswith('.jpg'))
  remote_cmd = f'rm {remotepath}'
  stdin, stdout, stderr = ssh_client.exec_command(remote_cmd)

def get_image_list(ssh_client, remotepath):
  image_list = []
  remote_cmd = f'ls {remotepath}'


  print(f'Remote cmd = <{remote_cmd}>')
  stdin, stdout, stderr = ssh_client.exec_command(remote_cmd)
  # We iterate over stdout
  for line in stdout:
    line = line.strip()
#    print('<-' + line + '->')  
    if line.endswith('.jpg'):
#      print('... ' + line)  
      image_list.append(line)
  return image_list
  


def get_images_iteration(ssh, sftp, remotepath, localpath):
  image_names = get_image_list(ssh, remotepath)
 
  #wait for the last image to be fully written
  time.sleep(3)  
  for img_name in image_names:
    remote_img_path = f'{remotepath}/{img_name}'
    local_img_path  = f'{localpath}/{int(datetime.datetime.now().timestamp())}_{img_name}'
    print(f'image_paths = {remote_img_path} --> {local_img_path}')
    get_image(sftp, localpath=local_img_path, remotepath=remote_img_path)
    del_remote_image(ssh, remote_img_path)




if __name__ == "__main__":
  remotepath='Documents/yoav/images'
  localpath=r'c:\temp'
  server='moratt'
  username='ymoratt'
  password='Sheshet7'

  print(f'getting images from {username}@{server}:{remotepath}')
  ssh, sftp = open_connection(server='moratt',username='ymoratt', password=password)
  
  for i in range(0,1000):
    print(f'Iteration {i}')
    get_images_iteration(ssh=ssh, sftp=sftp, remotepath=remotepath, localpath=localpath)
    time.sleep(10)


  close_connection(ssh, sftp)

  print('End')