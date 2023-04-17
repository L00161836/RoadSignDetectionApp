
#if ANDROID
using Android.Graphics;
using Android.Util;
using Java.IO;
using Java.Nio;
using Java.Nio.Channels;
using Java.Util;
using System.Collections.Generic;
using System.Drawing;
#endif

namespace RoadSignDetectionApp.Model
{
    // NB: Some of the code from this tutorial https://devblogs.microsoft.com/xamarin/image-classification-xamarin-android/ was used and adapted for this class.
    // This class functions as the interface for the application to the TF Lite model.
    public class TensorFlowClassifier
    {
        const int FloatSize = 4;
        const int PixelSize = 3;

        // This function takes an input image and runs it against the model, returning a SignClassificationModel object with sign probabilities.
        public static List<SignClassificationModel> Classify(ByteBuffer image)   
        {
            var mappedByteBuffer = GetModelAsMappedByteBuffer();
            var interpreter = new Xamarin.TensorFlow.Lite.Interpreter(mappedByteBuffer);

            var tensor = interpreter.GetInputTensor(0);
            var shape = tensor.Shape();

            var width = shape[1];
            var height = shape[2];

            var byteBuffer = GetResizedByteBuffer(image, height, width);

            var streamReader = new StreamReader(Android.App.Application.Context.Assets.Open("labels.txt"));

            var labels = streamReader.ReadToEnd().Split('\n').Select(s => s.Trim()).Where(s => !string.IsNullOrEmpty(s)).ToList();

            var outputLocations = new float[1][] { new float[labels.Count] };
            var outputs = Java.Lang.Object.FromArray(outputLocations);

            interpreter.Run(byteBuffer, outputs);
            var result = outputs.ToArray<float[]>();

            var modelList = new List<SignClassificationModel>();

            for (var i = 0; i < labels.Count; i++)
            {
                var label = labels[i];
                modelList.Add(new SignClassificationModel(label, result[0][i]));
            }

            return modelList;
            
        }

        //Brings the TF Lite model into the project as a mapped byte buffer.
        private static MappedByteBuffer GetModelAsMappedByteBuffer()
        {
            var assetDescriptor = Android.App.Application.Context.Assets.OpenFd("model.tflite");
            var inputStream = new FileInputStream(assetDescriptor.FileDescriptor);

            var mappedByteBuffer = inputStream.Channel.Map(FileChannel.MapMode.ReadOnly, assetDescriptor.StartOffset, assetDescriptor.DeclaredLength);

            return mappedByteBuffer;
        }

        /* Resizes the input image byte array to the correct size for the model.
         Commented out code shows an attempt to do this by converting the byte buffer to a bitmap and scaling that, however
         this returned multiple errors, especially with the BitmapFactory class. */
        private static ByteBuffer GetResizedByteBuffer(ByteBuffer image, int width, int height)
        {
            //BitmapFactory.Options options = new BitmapFactory.Options();
            //options.InJustDecodeBounds = true;
            //var bitmap = BitmapFactory.DecodeByteArray(image, 0, image.Length, options);
            //var resizedBitmap = Bitmap.CreateScaledBitmap(bitmap, width, height, true);

            var modelInputSize = FloatSize * height * width;
            var byteBuffer = ByteBuffer.AllocateDirect(modelInputSize);
            byteBuffer.Order(ByteOrder.NativeOrder());

            var pixel = 0;

            for (var i = 0; i < width; i++)
            {
                for (var j = 0; j < height; j++)
                {
                    byteBuffer.PutFloat(image.GetFloat(pixel));
                    pixel++;
                }
            }
            return byteBuffer;

            
        }
    }
}
