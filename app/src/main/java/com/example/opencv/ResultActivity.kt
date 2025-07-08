package com.example.opencv

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class ResultActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)

        val resultText: TextView = findViewById(R.id.textResult)

        val scanResult = intent.getStringExtra("SCAN_RESULT") ?: "Не удалось распознать код"
        resultText.text = scanResult
    }
}