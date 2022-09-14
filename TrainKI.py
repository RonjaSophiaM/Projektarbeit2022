# Ronja Maduch
# Projektarbeit "Erforschung einer intelligenten Android-App mit dem Zweck der Bilderkennung von Hautkrebs", 2022

# Es wird eine Künstliche Intelligenz zur Klassifizierung von gutartigen und
# bösartigen Hauterkrankungen aufgesetzt.

if __name__ == '__main__':

    # ---------- Importieren notwendiger Bibliotheken ----------

    import os
    import tensorflow as tf
    from tensorflow.keras.optimizers import RMSprop
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import matplotlib.pyplot as plt

    # ---------- Dateipfad der Daten definieren ----------

    base_dir = 'D:\Melanoma_Classification'

    # Dateipfad zu den Daten zum Trainieren der KI
    train_dir = os.path.join(base_dir, 'Training_Data')

    # Dateipfad zu den Daten zum Testen der KI
    validation_dir = os.path.join(base_dir, 'Test_Data')

    # Dateipfad zu den Daten zum Trainieren: gutartige Hauterkrankungen
    train_benign_dir = os.path.join(train_dir, 'NotMalignant')

    # Dateipfad zu den Daten zum Trainieren: bösartige Hauterkrankungen
    train_malignant_dir = os.path.join(train_dir, 'Malignant')

    # Dateipfad zu den Daten zum Testen: gutartige Hauterkrankungen
    validation_benign_dir = os.path.join(validation_dir, 'NotMalignant')

    # Dateipfad zu den Daten zum Testen: bösartige Hauterkrankungen
    validation_malignant_dir = os.path.join(validation_dir, 'Malignant')

    # ---------- Aufsetzen des Modells ----------

    # Es wird ein Convolutional Neural Network verwendet. Dieses hat 4 convolutional layers (gefaltete Schichten)
    # mit je 32, 64, 128 und 128 Faltungen. Anschließend wird das Flattening auf die daten angewendet, sodass die
    # Daten in einem tiefen neuronalen Netz analysiert und kategorisiert werden können.
    # Da es sich um ein Klassifizierungsproblem mit 2 Klassen handelt, wird die Sigmoidfunktion als
    # Aktivierungsfunktion verwendet.

    model = tf.keras.models.Sequential([
        # erste Faltungsschicht: Die Daten werden im Format 224x224 eingegeben und enthalten Informationen über
        # die Farben Rot Grün und Gelb. Es werden 32 Faltungen angewendet. Nach der Faltungsschicht folgt eine
        # Pooling-Schicht.
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # zweite Faltungsschicht mit 64 Faltungen und anschließender Pooling-Schicht
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # dritte Faltungsschicht mit 128 Faltungen und anschließender Pooling-Schicht
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # vierte Faltungsschicht mit 128 Faltungen und anschließender Pooling-Schicht
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Implementierung des Dropouts. Dies soll Overfitting vorbeugen.
        tf.keras.layers.Dropout(0.4),
        # Die Ergebnisse der Faltungen werden im Rahmen des flattenings in einen Vektor umgewandelt, damit die
        # Daten in einem tiefen neuronalen netzwerk verarbeitet werden können.
        tf.keras.layers.Flatten(),
        # Aufsetzen eines tiefen neuronalen Netzwerkes mit 512 Neuronen
        tf.keras.layers.Dense(512, activation='relu'),
        # Aufsetzen eines einzelnen Neurons der Ausgabeschicht. Es wird ein Wert zwischen 0 und 1 ausgegeben, wobei
        # Null und Eins jeweils für eine Klassifizierung stehen.
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Eine Zusammenfassung über das Aufgesetzte Modell wird ausgegeben
    model.summary()

    # Modell kompilieren
    model.compile(loss='binary_crossentropy',
    optimizer=RMSprop(learning_rate=0.001),
    metrics=['acc'])

    # ------------ Vorbereitung der Daten ------------

    # Skalierung der Bilder
    # Vorverarbeitung der Bilder durch Veränderung der Pixelwerte aus
    # dem Bereich [0, 254] in den Bereich [0, 1].
    # Alle Bilder werden um 1./255 skaliert.

    train_datagen = ImageDataGenerator(rescale=1. / 255)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Trainings und Testdaten in Stapel (Batches) einteilen
    train_generator = train_datagen.flow_from_directory(
        train_dir,  # Dateipfad der Trainingsdaten
        target_size=(224, 224), # Größe der Bilder
        batch_size=64,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir, # Dateipfad der Testdaten
        target_size=(224, 224), # Größe der Bilder
        batch_size= 64,
        class_mode='binary')

    print("train generator: ", train_generator.class_indices)
    print("validation generator: ", validation_generator.class_indices)

    # ------------ Training ------------

    history = model.fit(
        train_generator,
        steps_per_epoch=233,    # training_images / batch_size = steps_per_epoch ---> 14.944 / 64 = 233
        epochs=100,
        validation_data=validation_generator,
        validation_steps=58,    # testing_images / batch_size = steps_per_epoch ---> 3.736 / 64 = 58
        verbose=2)

    # ------------ Berechnung und Darstellung von Accuracy und Loss ------------
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    # ------------ Speichern des Modells ------------
    base_path = './CancerDetectionModel'
    model_path = os.path.join(base_path, 'my_model_cancer_detection_1')
    model.save(model_path)

    plt.show()

    print("- - -  ready - - -")
