import os
import paramiko


def get_image(server, username, password, localpath, remotepath):
  ssh = paramiko.SSHClient() 
  ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
  ssh.connect(server, username=username, password=password)
  sftp = ssh.open_sftp()
  # sftp.put(localpath, remotepath)
  sftp.get(localpath=localpath, remotepath=remotepath)
  sftp.close()
  ssh.close()


if __name__ == "__main__":
  print('Yey')

  get_image(server='moratt',username='ymoratt', password='Sheshet7',localpath=r'c:\temp\image_15.jpg', remotepath='Documents/yoav/images/image_15.jpg')

  print('End')