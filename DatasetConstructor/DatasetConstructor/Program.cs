// See https://aka.ms/new-console-template for more information
using DatasetConstructor.Saxotrader;

string token = "";

var saxoDataHandler = new SaxoDataHandler(token);

Console.WriteLine(await saxoDataHandler.GetUserData());