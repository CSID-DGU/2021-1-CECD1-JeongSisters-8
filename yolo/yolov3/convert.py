import os,re
os.chdir('/content/drive/MyDrive/YOLO_Stain_Tilda/darknet/data/obj/stain')
directory = os.listdir('./')
for file in directory:
    if file.endswith(".txt"):
        open_file = open(file,'r')
        new_line=''
        while True:
            string = open_file.readline()
            if not string:
                break
            new = open(file,'w')
            new_line += string.replace(string,"6"+string[1:])
        new.write(new_line)
        open_file.close()
        new.close()