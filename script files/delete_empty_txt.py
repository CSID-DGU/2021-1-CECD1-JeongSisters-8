import os,re
os.chdir('./img')
directory = os.listdir('./')
for file in directory:
    if file.endswith(".txt"):
        # print(file)
        open_file = open(file,'r')
        string = open_file.readline()
        if not string:
            jpg = file.replace('.txt','.jpg')
            os.remove(file)
            os.remove(jpg)
  