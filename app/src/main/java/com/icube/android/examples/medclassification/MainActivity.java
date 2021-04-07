package com.icube.android.examples.medclassification;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        TextView tv = findViewById(R.id.textView);
        Button btn = findViewById(R.id.button);
        ImageView imgv = findViewById(R.id.imageView);
        Integer resourceBitmap = R.drawable.anapril;
        imgv.setImageResource(resourceBitmap);
        Bitmap bm = BitmapFactory.decodeResource(getResources(), resourceBitmap);

        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String res = ClassificationService.processImage(v.getContext(), bm);
                tv.setText(res);
            }
        });
    }

}