package com.example.sdk

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Base64
import androidx.fragment.app.Fragment
import com.chaquo.python.Python
import com.example.sdk.R
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.ByteArrayOutputStream
import java.math.BigDecimal
import java.nio.ByteBuffer
import java.nio.ByteOrder
import com.example.sdk.ml.ModelScatDwtharrSiameseV1

class MainActivity : AppCompatActivity() {

        override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        //setContentView(R.layout.activity_main)
    }
//    fun processImage(key: Int, greymap: HashMap<Any, Bitmap?>): FloatArray {
//        var anc_enc = FloatArray(128)
//
//        var btmp = preprocessImage(key, greymap) //Function to preprocess the captured cropped Image
//
//        if (btmp != null) {
//            anc_enc =
//                ml_model(btmp) //This function sends the image inside the ml-model to get the desired features.
//        }
//        return anc_enc
//    }
//
//
//    fun getStringImage(grayBitmap: Bitmap?): String? {
//        var baos = ByteArrayOutputStream()
//        grayBitmap?.compress(Bitmap.CompressFormat.PNG, 100, baos)
//        var imgByte = baos.toByteArray()
//        var encodedImg = android.util.Base64.encodeToString(imgByte, android.util.Base64.DEFAULT)
//        return encodedImg
//    }
//
//    fun preprocessImage(key: Int, greymap: HashMap<Any, Bitmap?>): Bitmap? {
//
//        var py = Python.getInstance()
//        var pyObj = py.getModule("myscript")
//        var imagestr = getStringImage(greymap[key])
//        var obj = pyObj.callAttr("main", imagestr, 3)
//        var imgstr = obj.toString()
//        var data = Base64.decode(imgstr, Base64.DEFAULT)
//        var btmp = BitmapFactory.decodeByteArray(data, 0, data.size)
//        return btmp
//    }
//
//    fun ml_model(btmp: Bitmap): FloatArray {
//        var imageSize = 200
//        var py = Python.getInstance()
//
//        var pyObj = py.getModule("myscript")
//
//        var imagestr = getStringImage(btmp)
//        var obj = pyObj.callAttr("getpixel", imagestr)
//        var imgstr = obj.toString()
//        val intValues = imgstr.split(" ")
//
//        val inputFeature0 =
//            TensorBuffer.createFixedSize(intArrayOf(1, imageSize, imageSize), DataType.FLOAT32)
//        var byteBuffer1: ByteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize)
//
//        byteBuffer1.order(ByteOrder.nativeOrder())
//        var pixel = 0
//        for (i in 0 until 200) {
//            for (j in 0 until 200) {
////                        for(k in 0 until 3)
//                //                                    for (k in 0 until 3){
//                var vals = intValues[pixel++].toInt()// RGB
//                byteBuffer1.putFloat(
//                    (vals * (1F / 255f).toDouble().toBigDecimal()
//                        .setScale(6, BigDecimal.ROUND_HALF_UP).toFloat())
//                )
//            }
//        }
//
//        val model = ModelScatDwtharrSiameseV1.newInstance(requireActivity()) //input_image(200,200)
//        inputFeature0.loadBuffer(byteBuffer1)
//        val outputs = model.process(inputFeature0)
//        val outputFeature0 = outputs.outputFeature0AsTensorBuffer
//        var confidence = outputFeature0.floatArray
//
//        model.close()
//        return confidence
//    }
}