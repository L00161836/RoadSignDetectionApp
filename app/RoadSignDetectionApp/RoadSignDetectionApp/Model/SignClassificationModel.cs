using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RoadSignDetectionApp.Model
{
    public class SignClassificationModel
    {
        public string SignName { get; }
        public float Probability { get; }
        public SignClassificationModel(string signName, float probability)
        {
            SignName = signName;
            Probability= probability;
        }
    }
}
