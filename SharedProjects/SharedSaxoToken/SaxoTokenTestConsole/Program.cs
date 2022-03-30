


using SharedSaxoToken;

var selenium = new SeleniumDriver();
var token = "??";

var username = "";
var password = "";
var edgeLocation = "";

try
{
    token = await selenium.GetToken(username, password, edgeLocation);
}
catch (Exception e)
{
    Console.WriteLine("ERROR");
    throw;
}

Console.WriteLine("TOKEN:");
Console.WriteLine(token);


