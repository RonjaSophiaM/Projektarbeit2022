package com.ronjasophiamaduch.skincancerdetector

import android.content.ContentValues.TAG
import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Button

class overview_and_info : AppCompatActivity() {

    private val btn_to_main : Button by lazy{ findViewById(R.id.button_mainActivity) }
    private val btn_to_tutorial : Button by lazy{ findViewById(R.id.button_tutorial) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_overview_and_info)

        btn_to_main.setOnClickListener{
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
            Log.i(TAG, "btn_to_main clicked")
        }

        btn_to_tutorial.setOnClickListener{
            val intent = Intent(this, Tutorial::class.java)
            startActivity(intent)
            Log.i(TAG, "btn_to_tutorial clicked")
        }
    }
}