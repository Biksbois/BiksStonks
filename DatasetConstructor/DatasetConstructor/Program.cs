﻿
using DatasetConstructor.Saxotrader;
using DatasetConstructor.Saxotrader.Models;
using Microsoft.Extensions.Configuration;
using Newtonsoft.Json;
using SharedDatabaseAccess;
using SharedObjects.MetaTables;
using System;
using System.Globalization;

string token = GetAttributeFromConfig("token");

//var connectionString = GetAttributeFromConfig("ConnectionString");

//var stocks = new List<PriceValues>() { 
//    new PriceValues() { Close=1.0, High=1.0, Interest=1.0, Low=1.0, Open=1.0, Time=new DateTime(2022, 10, 10), Volume=1.0 },
//    new PriceValues() { Close=1.0, High=1.0, Interest=1.0, Low=1.0, Open=1.0, Time=new DateTime(2022, 10, 10), Volume=1.0 },
//    new PriceValues() { Close=1.0, High=1.0, Interest=1.0, Low=1.0, Open=1.0, Time=new DateTime(2022, 10, 11), Volume=1.0 },
//};

//var companies = new List<Company>() {
//    new Company() { Currencycode="DKK", Assettype="Stock", Exchangeid="CSE", Summarytype="Instrument", Issuercountry="DK", Category="UNKNOWN", Description="NET", Groupid="1", Identifier=1, Primarylisting="1", Symbol="YEET"},
//    new Company() { Currencycode="DKK", Assettype="Stock", Exchangeid="CSE", Summarytype="Instrument", Issuercountry="DK", Category="UNKNOWN", Description="NORLYS", Groupid="1", Identifier=2, Primarylisting="1", Symbol="YEET"},
//    new Company() { Currencycode="DKK", Assettype="Stock", Exchangeid="CSE", Summarytype="Instrument", Issuercountry="DK", Category="UNKNOWN", Description="NET", Groupid="1", Identifier=1, Primarylisting="1", Symbol="YEET"}
//};

//var conn = new StonksDbConnection();
//conn.InsertStocks(stocks, connectionString, 100);
//conn.InsertCompanies(companies, connectionString, "iUNKNOWN");

var saxoDataHandler = new SaxoDataHandler(token);

var DanishStocks = await saxoDataHandler.GetCompanyData(Exchange.CSE, AssetTypes.Stock);
var DatesToCheck = CalcDatesToCheck().Select(x => x.ToString("yyyy - MM - ddTHH:mm: ss.ffffffZ", CultureInfo.InvariantCulture));
var results = new Dictionary<Stock,List<PriceValues>>();


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
            Thread.Sleep(100000);
            results[DanishStock].AddRange(await saxoDataHandler.GetHistoricData(AssetTypes.Stock, DanishStock.Identifier, date));
            continue;
        }
    }
    CreateFileForDataPoints(DanishStock, results[DanishStock]);
}

static string GetAttributeFromConfig(string attribute)
{
    var config = new ConfigurationBuilder()
    .SetBasePath(AppDomain.CurrentDomain.BaseDirectory)
    .AddUserSecrets<Program>()
    .Build();

    return config[attribute];
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

async Task CreateFileForDataPoints(Stock stock, List<PriceValues> prices) 
{
    string path = "C:/Users/Stonker69/Desktop/Stonks/"+ stock.Description;
    string stockText = JsonConvert.SerializeObject(stock);
    string pricesText = JsonConvert.SerializeObject(prices);
    System.IO.Directory.CreateDirectory(path);
    await File.WriteAllTextAsync(path+"/stock.json", stockText);
    await File.WriteAllTextAsync(path+"/prices.json", pricesText);
}