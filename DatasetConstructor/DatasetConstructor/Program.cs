using DatasetConstructor.Saxotrader;
using DatasetConstructor.Saxotrader.Models;
using Microsoft.Extensions.Configuration;
using System;

string token = GetXtraderCredentials();

var saxoDataHandler = new SaxoDataHandler(token);

//Console.WriteLine(await saxoDataHandler.GetUserData());
StockData DanishStocks = await saxoDataHandler.GetCompanyData(Exchange.CSE, AssetTypes.Stock);
DataPoints DataPoints = await saxoDataHandler.GetHistoricData(AssetTypes.Stock, DanishStocks.Data[0].Identifier);
var test = CalcDatesToCheck();
Console.WriteLine("test");

static string GetXtraderCredentials()
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