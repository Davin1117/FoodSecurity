package com.ideutopia.thefarmersapp;
import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class TFLiteHelper {
   //Implementasi Label Data
   private List<String> label;
   private  Interpreter tflite;
    //Data Gambar
    private  int imageSizeX;
    private  int imageSizeY;
    //Tensorflow
    private MappedByteBuffer tfliteModel;
    private TensorImage inputImage;
    private TensorBuffer outputData;
    private TensorProcessor processorData;

    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD = 1.0f;
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILTY_STD = 255.0f;

    private Activity context;
    TFLiteHelper(Activity context) {this.context = context;}

    void init(){
        try {
            Interpreter.Options opt = new Interpreter.Options();
            tflite = new Interpreter(loadmodelfile(context), opt);
        }catch (Exception e){
            e.printStackTrace();
        }
    }
    private TensorImage loadImage(final Bitmap bitmap){
        //loads bitmpa into a tensor
        inputImage.load(bitmap);
        int cropImageSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeWithCropOrPadOp(cropImageSize, cropImageSize))
                .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .add(getPreprocessNormalizeOp())
                .build();
        return imageProcessor.process(inputImage);
    }
    private MappedByteBuffer loadmodelfile(Activity activity) throws IOException {
        String MODEL_NAME = "tanah.tflite";
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_NAME);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startoffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startoffset, declaredLength);
    }
    void classifyImage(Bitmap bitmap){
        int imageTensorIndex = 0;
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape();
        imageSizeX = imageShape[1];
        imageSizeY = imageShape[2];
        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();

        int probabilityTensorIndex = 0;
        int[] probabilityShape = tflite.getOutputTensor(probabilityTensorIndex).shape();
        DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

        inputImage = new TensorImage(imageDataType);
        outputData = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);
        processorData = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();
        inputImage = loadImage(bitmap);
        tflite.run(inputImage.getBuffer(), outputData.getBuffer().rewind());
    }

    public List<String> showresult() {
        try{
            label = FileUtil.loadLabels(context, "tanah.txt");
        }catch (Exception e){
            e.printStackTrace();
            return null;
        }
        Map<String, Float> labeledProbability = new TensorLabel(label,
                processorData.process(outputData))
                .getMapWithFloatValue();
        float maxValueInMap = (Collections.max(labeledProbability.values()));
        List<String> result = new ArrayList<>();
        for (Map.Entry<String, Float> entry : labeledProbability.entrySet()){
            if (entry.getValue() == maxValueInMap){
                result.add(entry.getKey());
            }
        }
        return result;
    }
    private  TensorOperator getPreprocessNormalizeOp(){
        return  new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }
    private TensorOperator getPostprocessNormalizeOp(){
        return  new NormalizeOp(PROBABILITY_MEAN, PROBABILTY_STD);
    }
}
