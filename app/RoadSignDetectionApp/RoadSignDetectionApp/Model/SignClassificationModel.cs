using Android.Nfc.Tech;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RoadSignDetectionApp.Model
{
    public class SignClassificationModel
    {   /* NB: The code for this object was largely inspired by https://devblogs.microsoft.com/xamarin/image-classification-xamarin-android/ 
           This class creates on object holding a sign name with a respective probability, making extracting the model results easier.*/
        public string SignName { get; }
        public float Probability { get; }
        public SignClassificationModel(string signName, float probability)
        {
            SignName = signName;
            Probability= probability;
        }
    }
}
