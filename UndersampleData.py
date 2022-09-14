# Ronja Maduch
# Projektarbeit "Erforschung einer intelligenten Android-App mit dem Zweck der Bilderkennung von Hautkrebs", 2022

# Es wird Undersampling angewendet, sodass die Anzahl der Bilder gutartiger und bösartiger Diagnosen ausgeglichen wird.

if __name__ == '__main__':
    import os
    import random

    # undersample data in folder
    # delete random pictures
    # in the train folder are 2761 malignant pictures and 7889 benign pictures
    # we keep 120%
    # 2761 * 1,2 = 3.313,2
    # therefore 4567 pictures are deleted in the benign folder

    # ---------- undersample malignant training Daten ----------
    dir = r'D:\Melanoma_Classification\Training_Data\NotMalignant'
    # In dem Ordner sind 7472 bösartige Bilder vorhanden und in dem anderen Ordner sind
    # 12792 Bilder von gutartigen Diagnosen. Daher werden 5320 Bilder aus dem Ordner mit den gutartigen Bildern entfernt.
    number_to_delete = 5320
    pic_no = 1
    while pic_no <= number_to_delete:
        print("_______________________________")
        print(pic_no, "/", number_to_delete)
        filename = random.choice(os.listdir(dir))
        print("delete: ", filename)
        path = os.path.join(dir, filename)
        os.remove(path)
        pic_no = pic_no+1

    print("------------------------------------------------------")
    print("number of deleted pictures:")
    print(pic_no-1)
    print("folder:")
    print(dir)


    # ---------- undersample malignant Test Daten ----------
    dir = r'D:\Melanoma_Classification\Test_Data\NotMalignant'
    # In dem Ordner sind 1868 bösartige Bilder vorhanden und in dem anderen Ordner sind
    # 3199 Bilder von gutartigen Diagnosen. Daher werden 1331 Bilder aus dem Ordner mit den gutartigen Bildern entfernt.
    number_to_delete = 1331
    pic_no = 1
    while pic_no <= number_to_delete:
        print("_______________________________")
        print(pic_no, "/", number_to_delete)
        filename = random.choice(os.listdir(dir))
        print("delete: ", filename)
        path = os.path.join(dir, filename)
        os.remove(path)
        pic_no = pic_no+1

    print("------------------------------------------------------")
    print("number of deleted pictures:")
    print(pic_no-1)
    print("folder:")
    print(dir)

    import sys
    sys.exit()