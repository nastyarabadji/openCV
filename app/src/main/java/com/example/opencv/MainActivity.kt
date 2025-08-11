package com.example.opencv

import android.app.ProgressDialog
import android.content.ClipData
import android.content.Intent
import android.graphics.*
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.google.zxing.*
import com.google.zxing.common.HybridBinarizer
import kotlinx.coroutines.*
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import java.util.EnumMap
import kotlin.math.min
import android.content.ClipboardManager
import org.opencv.BuildConfig
import org.opencv.core.Point

class MainActivity : AppCompatActivity() {
    private lateinit var scanButton: Button
    private val reader = MultiFormatReader().apply {
        setHints(EnumMap<DecodeHintType, Any>(DecodeHintType::class.java).apply {
            put(DecodeHintType.TRY_HARDER, true)
            put(DecodeHintType.POSSIBLE_FORMATS, listOf(BarcodeFormat.DATA_MATRIX))
        })
    }

    private val galleryLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK) {
            result.data?.data?.let { uri ->
                processImageFromGallery(uri)
            }
        }
    }

    private val scope = CoroutineScope(Dispatchers.Main + Job())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "OpenCV не загружен")
            Toast.makeText(this, "OpenCV не загружен", Toast.LENGTH_SHORT).show()
        } else {
            Log.d("OpenCV", "OpenCV успешно загружен")
        }

        scanButton = findViewById(R.id.scanButton)
        scanButton.setOnClickListener {
            openGalleryForImage()
        }
    }

    private fun openGalleryForImage() {
        val intent = Intent(Intent.ACTION_PICK).apply {
            type = "image/*"
            putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("image/jpeg", "image/png"))
        }
        galleryLauncher.launch(intent)
    }

    private fun processImageFromGallery(uri: Uri) {
        Log.d("ImageProcessing", "Начало обработки изображения из галереи")
        val progressDialog = ProgressDialog(this).apply {
            setMessage("Обработка изображения...")
            setCancelable(false)
            show()
        }

        scope.launch {
            try {
                // Шаг 1: Загрузка и уменьшение изображения
                Log.d("ImageProcessing", "Загрузка изображения...")
                val bitmap = withContext(Dispatchers.IO) {
                    MediaStore.Images.Media.getBitmap(contentResolver, uri).let {
                        Log.d("ImageProcessing", "Исходный размер: ${it.width}x${it.height}")
                        resizeBitmap(it, 1024).also { resized ->
                            Log.d("ImageProcessing", "Уменьшенный размер: ${resized.width}x${resized.height}")
                        }
                    }
                }

                // Шаг 2: Попытка распознавания без обработки
                Log.d("Decoding", "Попытка распознавания без обработки")
                var result = withTimeoutOrNull(2000) {
                    withContext(Dispatchers.Default) {
                        enhancedDecodeDataMatrix(bitmap, "Без обработки")
                    }
                }

                // Шаг 3: Если не распознано - полная обработка
                if (result == null) {
                    Log.d("Decoding", "Стандартное распознавание не удалось, пробуем с обработкой")
                    result = withTimeoutOrNull(5000) {
                        withContext(Dispatchers.Default) {
                            Log.d("ImageProcessing", "Начало улучшенной обработки изображения")
                            val processedBitmap = enhancedPreprocessImage(bitmap)
                            enhancedDecodeDataMatrix(processedBitmap, "После обработки")
                        }
                    }
                }

                // Шаг 4: Обработка результата
                result?.let {
                    Log.d("Decoding", "Успешно распознано: $it")
                    showResult(it)
                } ?: run {
                    Log.d("Decoding", "DataMatrix не найден после всех попыток")
                    Toast.makeText(this@MainActivity, "DataMatrix не найден", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Log.e("MainActivity", "Ошибка обработки: ${e.message}", e)
                Toast.makeText(this@MainActivity, "Ошибка обработки изображения", Toast.LENGTH_SHORT).show()
            } finally {
                progressDialog.dismiss()
                Log.d("ImageProcessing", "Завершение обработки изображения")
            }
        }
    }

    private fun showResult(result: String) {
        val resultTextView = findViewById<TextView>(R.id.scanResult)
        resultTextView.text = "Результат: $result"

        // Делаем текст копируемым по клику
        resultTextView.setOnClickListener {
            val clipboard = getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
            val clip = ClipData.newPlainText("Scan result", result)
            clipboard.setPrimaryClip(clip)
            Toast.makeText(this, "Результат скопирован", Toast.LENGTH_SHORT).show()
        }

        // Автоматически копируем в буфер обмена
        val clipboard = getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
        val clip = ClipData.newPlainText("Scan result", result)
        clipboard.setPrimaryClip(clip)
    }

    private fun resizeBitmap(bitmap: Bitmap, maxSize: Int): Bitmap {
        val ratio = min(maxSize.toFloat() / bitmap.width, maxSize.toFloat() / bitmap.height)
        val width = (bitmap.width * ratio).toInt()
        val height = (bitmap.height * ratio).toInt()
        return Bitmap.createScaledBitmap(bitmap, width, height, true)
    }

    private fun enhancedPreprocessImage(bitmap: Bitmap): Bitmap {
        Log.d("ImageProcessing", "Начало улучшенной обработки изображения")
        val src = Mat(bitmap.height, bitmap.width, CvType.CV_8UC3).apply {
            Utils.bitmapToMat(bitmap, this)
        }

        // 1. Уменьшение шума (менее агрессивное для сохранения точек)
        val denoised = Mat()
        Photo.fastNlMeansDenoisingColored(src, denoised, 7f, 7f, 5, 15)
        Log.d("ImageProcessing", "Применено уменьшение шума")

        // 2. Увеличение контраста (альтернативный метод)
        val lab = Mat()
        Imgproc.cvtColor(denoised, lab, Imgproc.COLOR_BGR2Lab)

        val channels = ArrayList<Mat>()
        Core.split(lab, channels)

        // Гистограммная эквализация с ограничением контраста (CLAHE)
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(channels[0], channels[0])
        // В OpenCV для Android нет метода close() для CLAHE, поэтому просто оставляем как есть
        // Ресурсы будут освобождены сборщиком мусора

        Core.merge(channels, lab)
        Imgproc.cvtColor(lab, denoised, Imgproc.COLOR_Lab2BGR)
        Log.d("ImageProcessing", "Применено увеличение контраста (CLAHE)")

        // 3. Два варианта бинаризации - выбираем лучший результат
        val gray = Mat()
        Imgproc.cvtColor(denoised, gray, Imgproc.COLOR_BGR2GRAY)

        // Вариант 1: Адаптивная бинаризация для обычных DataMatrix
        val binary1 = Mat()
        Imgproc.adaptiveThreshold(
            gray, binary1, 255.0,
            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
            Imgproc.THRESH_BINARY,
            15,  // Уменьшенный размер блока для точечных паттернов
            2.0
        )

        // Вариант 2: Пороговая бинаризация для точечных DataMatrix
        val binary2 = Mat()
        Imgproc.threshold(
            gray, binary2,
            0.0, 255.0,
            Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
        )

        // Выбираем бинаризацию с большим количеством контуров
        val binary = if (countContours(binary1) > countContours(binary2)) binary1 else binary2
        Log.d("ImageProcessing", "Выбрана бинаризация: ${if (binary == binary1) "адаптивная" else "Otsu"}")

        // 4. Морфологическая обработка - закрытие для соединения точек
        val kernelSize = if (binary == binary1) 3.0 else 5.0  // Большее ядро для точечных паттернов
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(kernelSize, kernelSize))
        Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, kernel, Point(-1.0, -1.0), 2)
        Log.d("ImageProcessing", "Применена морфологическая обработка с ядром $kernelSize")

        val resultBitmap = Bitmap.createBitmap(binary.cols(), binary.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(binary, resultBitmap)

        if (BuildConfig.DEBUG) {
            saveBitmapToFile(resultBitmap, Bitmap.CompressFormat.PNG, 100)
        }

        return resultBitmap
    }

    private fun countContours(binary: Mat): Int {
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(
            binary.clone(), // Клонируем, чтобы не изменять исходное изображение
            contours,
            hierarchy,
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )
        return contours.size
    }

    private fun enhancedDecodeDataMatrix(bitmap: Bitmap, attemptName: String): String? {
        Log.d("Decoding", "Попытка распознавания ($attemptName)")

        // Дополнительные параметры для точечных DataMatrix
        val pointMatrixHints = EnumMap<DecodeHintType, Any>(DecodeHintType::class.java).apply {
            put(DecodeHintType.TRY_HARDER, true)
            put(DecodeHintType.PURE_BARCODE, true)
            put(DecodeHintType.POSSIBLE_FORMATS, listOf(BarcodeFormat.DATA_MATRIX))
        }

        return try {
            // Стандартное распознавание
            tryDecode(bitmap, pointMatrixHints)?.also {
                Log.d("Decoding", "Успешно распознано стандартным методом")
                return it
            }

            // Инвертированное распознавание
            tryDecode(invertColors(bitmap), pointMatrixHints)?.also {
                Log.d("Decoding", "Успешно распознано инвертированным методом")
                return it
            }

            // Дополнительная попытка с размытием для точечных матриц
            val blurred = blurForDottedMatrix(bitmap)
            tryDecode(blurred, pointMatrixHints)?.also {
                Log.d("Decoding", "Успешно распознано после размытия")
                return it
            }

            null
        } catch (e: Exception) {
            Log.e("Decoding", "Ошибка при распознавании ($attemptName): ${e.message}")
            null
        }
    }

    private fun blurForDottedMatrix(bitmap: Bitmap): Bitmap {
        Log.d("Decoding", "Применение размытия для точечной матрицы")
        val src = Mat(bitmap.height, bitmap.width, CvType.CV_8UC1).apply {
            Utils.bitmapToMat(bitmap, this)
        }

        // Размытие для соединения точек
        val blurred = Mat()
        Imgproc.GaussianBlur(src, blurred, Size(3.0, 3.0), 0.0)

        // Повторная бинаризация
        val binary = Mat()
        Imgproc.threshold(
            blurred, binary,
            0.0, 255.0,
            Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU
        )

        return Bitmap.createBitmap(binary.cols(), binary.rows(), Bitmap.Config.ARGB_8888).apply {
            Utils.matToBitmap(binary, this)
        }
    }

    private fun tryDecode(bitmap: Bitmap, hints: Map<DecodeHintType, *>? = null): String? {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val source = RGBLuminanceSource(bitmap.width, bitmap.height, pixels)
        val binaryBitmap = BinaryBitmap(HybridBinarizer(source))

        // Создаем временный reader с нужными hints
        val tempReader = MultiFormatReader().apply {
            hints?.let { setHints(it) }
        }

        return try {
            tempReader.decode(binaryBitmap)?.text
        } catch (e: NotFoundException) {
            null
        } catch (e: Exception) {
            Log.e("Decoding", "Ошибка в tryDecode: ${e.message}")
            null
        } finally {
            tempReader.reset()
        }
    }

    private fun saveBitmapToFile(bitmap: Bitmap, format: Bitmap.CompressFormat, quality: Int): File? {
        return try {
            val storageDir = File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
                "MyProcessedImages"
            ).apply { mkdirs() }

            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val fileName = "processed_${timeStamp}.${format.name.lowercase(Locale.ROOT)}"
            File(storageDir, fileName).apply {
                FileOutputStream(this).use { out ->
                    bitmap.compress(format, quality, out)
                    out.flush()
                }
            }
        } catch (e: Exception) {
            null
        }
    }

    private fun tryDecode(bitmap: Bitmap): String? {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val source = RGBLuminanceSource(bitmap.width, bitmap.height, pixels)
        val binaryBitmap = BinaryBitmap(HybridBinarizer(source))

        return try {
            reader.decode(binaryBitmap)?.text
        } catch (e: NotFoundException) {
            null
        } catch (e: Exception) {
            Log.e("Decoding", "Ошибка в tryDecode: ${e.message}")
            null
        }
    }

    private fun invertColors(bitmap: Bitmap): Bitmap {
        Log.d("Decoding", "Инвертирование цветов изображения")
        return bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, true).apply {
            val canvas = Canvas(this)
            val paint = Paint().apply {
                colorFilter = ColorMatrixColorFilter(ColorMatrix(floatArrayOf(
                    -1f, 0f, 0f, 0f, 255f,
                    0f, -1f, 0f, 0f, 255f,
                    0f, 0f, -1f, 0f, 255f,
                    0f, 0f, 0f, 1f, 0f
                )))
            }
            canvas.drawBitmap(bitmap, 0f, 0f, paint)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel() // Отменяем все корутины при уничтожении Activity
    }
}