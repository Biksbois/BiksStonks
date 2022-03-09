
using DatasetConstructor.Saxotrader;
using DatasetConstructor.Saxotrader.Models;
using Microsoft.Extensions.Configuration;
using Newtonsoft.Json;
using SharedDatabaseAccess;
using SharedObjects.MetaTables;
using System;
using System.Linq;
using System.Globalization;
using DatasetConstructor;



var config = GetConfig();

var token = config["token"];
var connectionString = config["ConnectionString"];
var dataFolder = config["datafolder"];

ConstructDataset constructDataset = new ConstructDataset(token, connectionString);

var companies = new List<string>() { "Danske Bank A/S", "Vestas Wind Systems A/S" };

await constructDataset.ScrapeDataToFolder(dataFolder, companies);

constructDataset.InsertDatafolder(dataFolder);

static IConfigurationRoot? GetConfig()
{
    var config = new ConfigurationBuilder()
    .SetBasePath(AppDomain.CurrentDomain.BaseDirectory)
    .AddUserSecrets<Program>()
    .Build();

    return config;
}


