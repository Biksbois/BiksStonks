using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DatasetConstructor.Saxotrader.Models
{
    public class Stock
    {
        public string AssetType { get; set; }
        public string CurrencyCode { get; set; }
        public string Description { get; set; }
        public string ExchangeId { get; set; }
        public int GroupId { get; set; }
        public int Identifier { get; set; }
        public string IssuerCountry { get; set; }
        public int PrimaryListing { get; set; }
        public string SummaryType { get; set; }
        public string Symbol { get; set; }
        public List<string> TradableAs { get; set; }
        public string Category { get; set; }
    }
}
