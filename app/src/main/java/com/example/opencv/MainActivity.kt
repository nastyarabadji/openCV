package com.example.opencv

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import com.journeyapps.barcodescanner.ScanContract
import com.journeyapps.barcodescanner.ScanOptions

class MainActivity : AppCompatActivity() {
    private lateinit var scanButton: Button

    private val dataMatrixLauncher = registerForActivityResult(ScanContract()) { result ->
        if (result.contents != null) {
            // Intent для перехода в ResultActivity
            val intent = Intent(this, ResultActivity::class.java).apply {
                putExtra("SCAN_RESULT", result.contents)
            }
            startActivity(intent)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        scanButton = findViewById(R.id.scanButton)

        scanButton.setOnClickListener {
            val scanOptions = ScanOptions().apply {
                setDesiredBarcodeFormats(ScanOptions.DATA_MATRIX)
                setPrompt("")
                setCameraId(0)
                setBeepEnabled(false)
                setOrientationLocked(true)
                setBarcodeImageEnabled(true)
            }
            dataMatrixLauncher.launch(scanOptions)
        }
    }
}