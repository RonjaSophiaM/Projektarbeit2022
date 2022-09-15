package com.ronjasophiamaduch.skincancerdetector

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.isVisible
import com.ronjasophiamaduch.skincancerdetector.ml.MyTfliteModelCancerDetection2
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder


class MainActivity : AppCompatActivity() {

    private val imageView : ImageView by lazy{ findViewById(R.id.imageView) }
    private val btnLoad : Button by lazy{ findViewById(R.id.btn_load_image) }
    private val btnPredict : Button by lazy{ findViewById(R.id.btn_predice_image) }
    private val tvOutput : TextView by lazy{ findViewById(R.id.tv_output) }
    private lateinit var scaledBitmap: Bitmap
    private var bitmapFlag = false
    private lateinit var mBitmap: Bitmap
    lateinit var model: MyTfliteModelCancerDetection2
    private val galleryRequestCode = 2
    private val mInputSize = 224

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Modell zur Klassifizierung laden
        model = MyTfliteModelCancerDetection2.newInstance(this)

        // Button für Vorhersagen
        btnPredict.setOnClickListener{
            // Wenn schon ein Bild geladen wurde...
            if(bitmapFlag){
                // Bild formatieren
                scaledBitmap = Bitmap.createScaledBitmap(scaledBitmap, 224, 224, false)
                val byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3)
                byteBuffer.order(ByteOrder.nativeOrder())
                // 1D Array mit Breite * Höhe Pixeln im Bild
                val intValues = IntArray(224 * 224)
                scaledBitmap.getPixels(intValues, 0, scaledBitmap.width, 0, 0, scaledBitmap.width, scaledBitmap.height)

                // Pixel durchlaufen und R-, G- und B-Werte extrahieren. Zum Bytebuffer hinzufügen.
                var pixel = 0
                for (i in 0 until 224) {
                    for (j in 0 until 224) {
                        val `val` = intValues[pixel++] // RGB
                        byteBuffer.putFloat((`val` shr 16 and 0xFF) * (1f / 255f))
                        byteBuffer.putFloat((`val` shr 8 and 0xFF) * (1f / 255f))
                        byteBuffer.putFloat((`val` and 0xFF) * (1f / 255f))
                    }
                }

                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
                if (byteBuffer != null) {
                    inputFeature0.loadBuffer(byteBuffer)
                    // Modell verwenden, um Daten zu klassifizierern
                    val outputs = model.process(inputFeature0)
                    // Ergebnis der Klassifizierung abrufen
                    val outputFeature0 = outputs.outputFeature0AsTensorBuffer
                    var prozentFloat = outputFeature0.getFloatValue(0)
                    Log.i("TAG", "Prozent_float: $prozentFloat")
                    // Ausgabe runden
                    prozentFloat = (prozentFloat + 0.005).toFloat()
                    var prozentInt = (prozentFloat * 100).toInt()
                    Log.i("TAG", "Prozentint: $prozentInt")
                    // weniger als 50 bedeutet, dass es eine bösartige Erkrankung sein könnte, ansonsten
                    // handelt es sich wahrscheinlich um eine gutartige Erkrankung
                    if(prozentInt < 50){
                        prozentInt = 100 - prozentInt
                        tvOutput.isVisible = true
                        tvOutput.text = getString(R.string.result_categorization_first_part) + prozentInt.toString() + getString(R.string.result_categorization_dangerous)
                        tvOutput.setTextColor(Color.parseColor("#A50A0A"))
                    }else{
                        tvOutput.isVisible = true
                        tvOutput.text = getString(R.string.result_categorization_first_part) + prozentInt.toString() + getString(R.string.result_categorization_harmless)
                        tvOutput.setTextColor(Color.parseColor("#057C05"))
                    }
                }
            } else{
                // Wenn noch kein Foto zur Klassifizierung hochgeladen wurde: Toas anzeigen
                Toast.makeText(this, R.string.please_load_picture_toast, Toast.LENGTH_SHORT).show()
            }
        }

        // Abbildung aus der Gallery laden
        btnLoad.setOnClickListener{
            Log.i("TAG", "btn_load was clicked")

            val callGalleryIntent = Intent(Intent.ACTION_PICK)
            callGalleryIntent.type = "image/*"
            startActivityForResult(callGalleryIntent, galleryRequestCode)
            bitmapFlag = true
        }
    }

    // Die Instanz der Modells der künstlichen Intelligenz wird gelöscht, wenn die
    // Seite geschlossen wird.
    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }


    // Wenn eine Abbildung aus der Gallerie ausgewählt wurde: als Bitmap formatieren
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (data != null) {
            val uri = data.data
            try {
                mBitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            } catch (e: IOException) {
                e.printStackTrace()
            }

            val orignalWidth = mBitmap!!.width
            val originalHeight = mBitmap.height
            val scaleWidth = mInputSize.toFloat() / orignalWidth
            val scaleHeight = mInputSize.toFloat() / originalHeight
            val matrix = Matrix()
            matrix.postScale(scaleWidth, scaleHeight)
            scaledBitmap = Bitmap.createBitmap(mBitmap, 0, 0, orignalWidth, originalHeight, matrix, true)

            imageView.setImageBitmap(mBitmap)
        }
    }

}

