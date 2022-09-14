package com.ronjasophiamaduch.skincancerdetector

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.ActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.isVisible
import com.ronjasophiamaduch.skincancerdetector.ml.MyTfliteModelCancerDetection2
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder


class MainActivity : AppCompatActivity() {

    private val imageView : ImageView by lazy{ findViewById(R.id.imageView) }
    private val btnLoad : Button by lazy{ findViewById(R.id.btn_load_image) }
    private val btnPredict : Button by lazy{ findViewById(R.id.btn_predice_image) }
    private val tvOutput : TextView by lazy{ findViewById(R.id.tv_output) }
    private val GALLERY_REQUEST_CODE = 123
    private lateinit var scaledBitmap: Bitmap
    private var bitmapFlag = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnPredict.setOnClickListener{
            if(bitmapFlag){
                scaledBitmap = Bitmap.createScaledBitmap(scaledBitmap, 224, 224, false)
                val bitmap = scaledBitmap.copy(Bitmap.Config.ARGB_8888, true)
                val byteBuffer = bitmapToByteBuffer(bitmap, 224, 224)
                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
                if (byteBuffer != null) {
                    inputFeature0.loadBuffer(byteBuffer)
                    // Modell zur Klassifizierung laden
                    val model = MyTfliteModelCancerDetection2.newInstance(this)
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

                    model.close()
                }

            } else{
                // Wenn noch kein Foto zur Klassifizierung hochgeladen wurde: Toas anzeigen
                Toast.makeText(this, R.string.please_load_picture_toast, Toast.LENGTH_SHORT).show()
            }

        }

        btnLoad.setOnClickListener{
            Log.i("TAG", "btn_load was clicked")
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            //intent.type = "image/*"
            val mimeTypes = arrayListOf("image/jpeg","image/png","image/jpg")
            intent.putExtra(Intent.EXTRA_MIME_TYPES, mimeTypes)
            intent.flags = Intent.FLAG_GRANT_READ_URI_PERMISSION
            onresult.launch(intent)
        }
    }

    private fun bitmapToByteBuffer(image: Bitmap, width: Int, height: Int): ByteBuffer? {
        val byteBuffer = ByteBuffer.allocateDirect(4 * width * height * 3)
        byteBuffer.order(ByteOrder.nativeOrder())
        // 1D Array mit Breite * Höhe Pixeln im Bild
        val intValues = IntArray(width * height)
        image.getPixels(intValues, 0, image.width, 0, 0, image.width, image.height)

        // Pixel durchlaufen und R-, G- und B-Werte extrahieren. Zum Bytebuffer hinzufügen.
        var pixel = 0
        for (i in 0 until width) {
            for (j in 0 until height) {
                val `val` = intValues[pixel++] // RGB
                byteBuffer.putFloat((`val` shr 16 and 0xFF) * (1f / 255f))
                byteBuffer.putFloat((`val` shr 8 and 0xFF) * (1f / 255f))
                byteBuffer.putFloat((`val` and 0xFF) * (1f / 255f))
            }
        }
        return byteBuffer
    }


    // Bild aus der Galery
    private val onresult = registerForActivityResult(ActivityResultContracts.StartActivityForResult()){result->
        Log.i("TAG", "This is the result: ${result.data} ${result.resultCode}")
        onResultReceived(GALLERY_REQUEST_CODE,result)
        tvOutput.isVisible = false
    }

    private fun onResultReceived(requestCode: Int, result: ActivityResult){
        when(requestCode){
            GALLERY_REQUEST_CODE->{
                if(result.resultCode == Activity.RESULT_OK){
                    result.data?.data.let { uri-> //result.data?.data?.let{
                        Log.i("TAG", "on result received: $uri")
                        val bitmap = BitmapFactory.decodeStream(contentResolver.openInputStream(uri!!))
                        imageView.setImageBitmap(bitmap)
                        bitmapFlag = true
                        scaledBitmap = bitmap
                        tvOutput.isVisible = false
                    }
                }else {
                    Log.e("TAG", "onActivityResult: error in selecting image")
                }
            }
        }
    }
}


