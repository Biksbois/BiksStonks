
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
using SharedSaxoToken;

var config = GetConfig();

var username = config["saxoUsername"];
var password = config["saxoPassword"];
var edgeLocation = config["edgeLocation"];

var yearsBack = 10;

var connectionString = config["ConnectionString"];
var dataFolder = config["datafolder"];

// **See ESG:xcse(Ennogie Solar Group A/S)

ConstructDataset constructDataset = new ConstructDataset(connectionString, username, password, edgeLocation);

var companies = new List<string>() { "Danske Bank A/S", "Vestas Wind Systems A/S" };

//await constructDataset.ScrapeDataToFolder(dataFolder, yearsBack);

constructDataset.InsertDatafolder(dataFolder);

static IConfigurationRoot? GetConfig()
{
    var config = new ConfigurationBuilder()
    .SetBasePath(AppDomain.CurrentDomain.BaseDirectory)
    .AddUserSecrets<Program>()
    .Build();

    return config;
}


