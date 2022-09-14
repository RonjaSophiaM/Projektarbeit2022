package com.ronjasophiamaduch.skincancerdetector

import android.content.ContentValues.TAG
import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Button

class OverviewAndInfo : AppCompatActivity() {

    private val btnToMain : Button by lazy{ findViewById(R.id.button_mainActivity) }
    private val btnToTutorial : Button by lazy{ findViewById(R.id.button_tutorial) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_overview_and_info)

        btnToMain.setOnClickListener{
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
            Log.i(TAG, "btnToMain clicked")
        }

        btnToTutorial.setOnClickListener{
            val intent = Intent(this, Tutorial::class.java)
            startActivity(intent)
            Log.i(TAG, "btnToTutorial clicked")
        }
    }
}