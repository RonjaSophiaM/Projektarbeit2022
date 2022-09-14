# Ronja Maduch
# Projektarbeit "Erforschung einer intelligenten Android-App mit dem Zweck der Bilderkennung von Hautkrebs", 2022

# Das Model wird zu einer .tflite Modell konvertiert.

if __name__ == '__main__':

    # Bibliotheken importieren
    import tensorflow as tf
    # Converting a SavedModel to a TensorFlow Lite model.
    # converter = tf.lite.TFLiteConverter.from_saved_model(r'D:\GithubRespiratory\CancerDetectionModel\my_model_cancer_detection_1')
    converter = tf.lite.TFLiteConverter.from_saved_model(r'D:\GithubRespiratory\CancerDetectionModel\my_model_cancer_detection_2')
    my_tflite_model = converter.convert()
    print(my_tflite_model)
    # Save the model.
    import pathlib

    # tflite_model_files = pathlib.Path('D:\GithubRespiratory\CancerDetectionModel\my_tflite_model_cancer_detection_1.tflite')
    tflite_model_files = pathlib.Path('D:\GithubRespiratory\CancerDetectionModel\my_tflite_model_cancer_detection_2.tflite')
    tflite_model_files.write_bytes(my_tflite_model)

    print("ready")
    import sys
    sys.exit()