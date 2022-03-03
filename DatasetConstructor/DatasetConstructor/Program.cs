using DatasetConstructor.Saxotrader;
using DatasetConstructor.Saxotrader.Models;
using Microsoft.Extensions.Configuration;
using System;
using System.Globalization;

string token = GetToken();

var saxoDataHandler = new SaxoDataHandler(token);

StockData DanishStocks = await saxoDataHandler.GetCompanyData(Exchange.CSE, AssetTypes.Stock);
var DatesToCheck = CalcDatesToCheck().Select(x => x.ToString("yyyy - MM - ddTHH:mm: ss.ffffffZ", CultureInfo.InvariantCulture));
Dictionary<Stock,List<PriceValues>> results = new Dictionary<Stock,List<PriceValues>>();
foreach (Stock DanishStock in DanishStocks.Data) 
{
    results.Add(DanishStock, new List<PriceValues>());
    Console.WriteLine();
    foreach (string date in DatesToCheck) 
    {
        try
        {
            results[DanishStock].AddRange(await saxoDataHandler.GetHistoricData(AssetTypes.Stock, DanishStock.Identifier, date));
        }
        catch (Exception)
        {
            System.Threading.Thread.Sleep(100000);
            results[DanishStock].AddRange(await saxoDataHandler.GetHistoricData(AssetTypes.Stock, DanishStock.Identifier, date));
            continue;
        }
    }
}

static string GetToken()
{
    var config = new ConfigurationBuilder()
    .SetBasePath(AppDomain.CurrentDomain.BaseDirectory)
    .AddUserSecrets<Program>()
    .Build();

    string token = config["token"];

    return token;
}

List<DateTime> CalcDatesToCheck(int years = 2,int Horizon = 1, int Count = 1200) 
{
    var To = DateTime.UtcNow;
    var From = To.AddYears(-years);
    double hours = ((Horizon * Count) / 60);
    List<DateTime> result = new List<DateTime>();
    double totalHours = (To - From).TotalHours;
    DateTime current = To;
    for (int i = 0; i < (totalHours/hours); i++)
    {
        current = current.AddHours(-hours);
        result.Add(current);
    }
    return result;
} 