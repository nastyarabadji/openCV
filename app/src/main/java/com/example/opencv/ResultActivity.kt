package com.example.opencv

import android.content.ClipData
import android.content.ClipboardManager
import android.os.Bundle
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

class ResultActivity : AppCompatActivity() {
    private lateinit var resultText: TextView
    private lateinit var toolbar: Toolbar

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_result)

        toolbar = findViewById(R.id.toolbarBack)
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Результат"

        resultText = findViewById(R.id.resultTextView)
        val scanResult = intent.getStringExtra("SCAN_RESULT") ?: "Не удалось распознать код"
        resultText.text = scanResult

        resultText.setOnClickListener {
            copyToClipboard(scanResult)
            Toast.makeText(this, "Результат скопирован", Toast.LENGTH_SHORT).show()
        }
    }

    private fun copyToClipboard(text: String) {
        val clipboard = getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
        val clip = ClipData.newPlainText("Scan result", text)
        clipboard.setPrimaryClip(clip)
    }

    override fun onSupportNavigateUp(): Boolean {
        onBackPressed()
        return true
    }
}