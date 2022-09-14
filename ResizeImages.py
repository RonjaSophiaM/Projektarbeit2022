# Ronja Maduch
# Projektarbeit "Erforschung einer intelligenten Android-App mit dem Zweck der Bilderkennung von Hautkrebs", 2022

# Die Bilder werden auf die Größe 224x224 skaliert.

if __name__ == '__main__':

    # Bibliotheken importieren
    import cv2
    import os

    def resize(directory_name, width, height):

        for file_name in os.listdir(directory_name):
            i = os.path.join(directory_name, file_name)

            if os.path.isfile(i):

                src = cv2.imread(i, cv2.IMREAD_UNCHANGED)

                dsize = (width, height)

                # Bildgröße zu 224x224 verändern
                output = cv2.resize(src, dsize)

                # Datei überschreiben
                cv2.imwrite(i, output)


    directory_name = r'D:\Melanoma_Classification\Training_Data\NotMalignant'
    resize(directory_name, width=224, height=224)

    directory_name = r'D:\Melanoma_Classification\Training_Data\Malignant'
    resize(directory_name, width=224, height=224)

    directory_name = r'D:\Melanoma_Classification\Test_Data\Malignant'
    resize(directory_name, width=224, height=224)

    directory_name = r'D:\Melanoma_Classification\Test_Data\NotMalignant'
    resize(directory_name, width=224, height=224)

    import sys
    sys.exit()