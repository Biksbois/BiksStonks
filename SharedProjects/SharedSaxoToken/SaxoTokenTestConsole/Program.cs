


using Microsoft.Extensions.Configuration;
using SharedDatabaseAccess;
using SharedSaxoToken;

var token = "??";

var config = GetConfig();

//var token = config["token"];
var username = config["saxoUsername"];
var password = config["saxoPassword"];
var edgeLocation = config["edgeLocation"];
var connectionString = config["connectionString"];

var stonksdb = new StonksDbConnection();

try
{
    token = await SaxoToken.GetAsync(username, password, edgeLocation, connectionString, stonksdb);
    //token = await SeleniumDriver.GetToken(username, password, edgeLocation);
}
catch (Exception e)
{
    Console.WriteLine("ERROR");
    throw;
}

Console.WriteLine("TOKEN:");
Console.WriteLine(token);

static IConfigurationRoot? GetConfig()
{
    var config = new ConfigurationBuilder()
    .SetBasePath(AppDomain.CurrentDomain.BaseDirectory)
    .AddUserSecrets<Program>()
    .Build();

    return config;
}
