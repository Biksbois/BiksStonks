using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DatasetConstructor.Saxotrader.Models
{
    // Root myDeserializedClass = JsonConvert.DeserializeObject<Root>(myJsonResponse);
    public class PriceValues
    {
        public double Close { get; set; }
        public double High { get; set; }
        public double Interest { get; set; }
        public double Low { get; set; }
        public double Open { get; set; }
        public DateTime Time { get; set; }
        public double Volume { get; set; }
    }

    public class DataPoints
    {
        public List<PriceValues> Data { get; set; }
        public int DataVersion { get; set; }
    }


}
