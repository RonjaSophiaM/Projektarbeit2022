# Ronja Maduch
# Projektarbeit "Erforschung einer intelligenten Android-App mit dem Zweck der Bilderkennung von Hautkrebs", 2022

# Das Model wird getestet.



if __name__ == '__main__':
    # Bibliotheken importieren
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    import os
    import random
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.pyplot as plt


    # Test funktion definieren
    def test(number_all_tests, dir, model, sizeX, sizeY):
        test_no = 1
        mal_counter = 0
        ben_counter = 0
        results = []
        print("Folder: ", dir)
        while test_no <= number_all_tests:
            print("-------------")
            print(test_no)
            # Zufällige Datei aus dem Ordner
            filename = random.choice(os.listdir(dir))  # change dir name to whatever
            print(filename)
            # Dateipfad der Datei
            path = os.path.join(dir, filename)
            # Daten vorbereiten
            img = load_img(path)
            img = img.resize((sizeX, sizeY))
            img = img_to_array(img)
            img = img / 255
            img = img.reshape(-1, sizeX, sizeY, 3)
            # Model zur klassifizierung verwenden
            classes = model.predict(img)
            # Ergebnis der Klassifizierung zur Liste hinzufügen
            results.append(classes[0])
            if classes[0] < 0.5:
                # Ergebnisse der Klassifitierung ausgeben
                print("It is malignant for " + str(float(classes[0]) * 100) + "%")
                # hochzählen
                mal_counter = mal_counter + 1
            else:
                prozent = 100.0 - (float(classes[0]) * 100)
                # Ergebnisse der Klassifitierung ausgeben
                print("It is benign for " + str(prozent) + "%")
                # hochzählen
                ben_counter = ben_counter + 1
            # hochzählen
            test_no = test_no + 1
        # Prozente der Testergebnisse berechnen
        p_mal = mal_counter / number_all_tests
        p_ben = ben_counter / number_all_tests
        # Durchschnitt der Testergebnisse berechnen
        average_results = sum(results) / len(results)
        return mal_counter, p_mal, ben_counter, p_ben, results, average_results


    # Dateipfade zu den Testdateien definieren
    dir_mal = r'D:\Melanoma_Classification\Test_Data\Malignant'
    dir_ben = r'D:\Melanoma_Classification\Test_Data\NotMalignant'
    #dir_ben = "D:\eigens_aufgenommene_Bilder_gutartig"
    # Nummer der Tests pro Ordner festlegen
    number_all_tests = 20
    # Modell laden
    #model = keras.models.load_model(r'D:\GithubRespiratory\CancerDetectionModel\my_model_cancer_detection_1')
    model = keras.models.load_model(r'D:\GithubRespiratory\CancerDetectionModel\my_model_cancer_detection_2')
    # Größe der Abbildungen definieren
    sizeX = 224
    sizeY = 224
    # Tests durchführen
    mal_counter1, p_mal1, ben_counter1, p_ben1, results_mal, average_results_mal = test(number_all_tests, dir_mal, model, sizeX, sizeY)
    print("\n\n\n------------------------------------------------------\n\n\n")
    mal_counter2, p_mal2, ben_counter2, p_ben2, results_ben, average_results_ben = test(number_all_tests, dir_ben, model, sizeX, sizeY)
    print("\n\n\n------------------------------------------------------")
    print("------------------------------------------------------")
    print("------------------------------------------------------\n\n\n")

    # Testergebnisse ausgeben

    print("number of tests per folder:")
    print(number_all_tests)

    print("\n\nfolder 1:", dir_mal)
    print("mal_counter1:", mal_counter1, "mal %", p_mal1)
    print("ben_counter1:", ben_counter1, "ben %", p_ben1)
    print("average: ", average_results_mal)

    print("\n\nfolder 2:", dir_ben)
    print("mal_counter2:", mal_counter2, "mal %", p_mal2)
    print("ben_counter2:", ben_counter2, "ben %", p_ben2)
    print("average: ", average_results_ben)

    print("\n\n---All in all result:---")
    percentage_right = (p_mal1 + p_ben2) / 2
    percentage_wrong = (p_ben1 + p_mal2) / 2
    print("right:", percentage_right)
    print("wrong:", percentage_wrong)

    # eine Abbildung erstellen, die die Verteilung der Testergebnisse zeigt
    fig, ax = plt.subplots()
    # Titel der Abbildung
    fig.suptitle('Distribution of test results', fontsize=16)
    # X-Achse
    test = np.arange(0, number_all_tests, 1)
    # Daten in das Streudiagramm eingeben
    ax.scatter(test, results_mal, color='red', label='malignant', s=1.5)
    ax.scatter(test, results_ben, color='green', label='benign', s=1.5)
    # Gitter
    ax.grid(True)
    # Durchschnitt zeigen
    y_avg_mal = [average_results_mal] * number_all_tests
    mean_line_mal = ax.plot(test, y_avg_mal, label='average_mal', linestyle='--', color='red')
    y_avg_ben = [average_results_ben] * number_all_tests
    mean_line_ben = ax.plot(test, y_avg_ben, label='average_ben', linestyle='--', color='green')
    # Titel der Y-Achse festlegen
    ax.set_ylabel('results')
    # Legende
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    # X-Achse verstecken
    ax.get_xaxis().set_visible(False)
    # Plot zeigen
    plt.show()

    import sys
    sys.exit()