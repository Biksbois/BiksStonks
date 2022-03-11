using DatasetConstructor.Saxotrader;
using DatasetConstructor.Saxotrader.Models;
using Newtonsoft.Json;
using SharedDatabaseAccess;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DatasetConstructor
{
    public class ConstructDataset
    {
        private readonly SaxoDataHandler _saxoDataHandler;
        private readonly string _connectionString;

        public ConstructDataset(string token, string connectionString)
        {
            _saxoDataHandler = new SaxoDataHandler(token);
            _connectionString = connectionString;
        }

        public void InsertDatafolder(string dataFolder)
        {
            var conn = new StonksDbConnection();

            foreach (var (company, stocks) in ExtractDataFromPath(dataFolder))
            {
                stocks.ForEach(stock => stock.Time = DateTime.SpecifyKind(stock.Time, DateTimeKind.Unspecified));

                conn.InsertStocks(stocks, _connectionString);
                conn.InsertCompanies(new List<Company>() { company }, _connectionString, "UNKNOWN");
            }
        }

        public async Task ScrapeDataToFolder(string dataFolder)
        {
            var DanishStocks = await _saxoDataHandler.GetAllCompanyData(Exchange.CSE, AssetTypes.Stock);

            DanishStocks.RemoveAll(x => x.Description.Contains("**See ESG:xcse(Ennogie Solar Group A/S)"));

            var a = DanishStocks.Select(stock => stock.Description).ToList();

            await ScrapeDataFromStocks(dataFolder, _saxoDataHandler, DanishStocks);
        }

        public async Task ScrapeDataToFolder(string dataFolder, List<string> companies)
        {
            var DanishStocks = await _saxoDataHandler.GetAllCompanyData(Exchange.CSE, AssetTypes.Stock);

            var relevantCompanies = DanishStocks.Where(x => companies.Contains(x.Description)).ToList();

            await ScrapeDataFromStocks(dataFolder, _saxoDataHandler, relevantCompanies);
        }

        private async Task ScrapeDataFromStocks(string dataFolder, SaxoDataHandler saxoDataHandler, List<Stock> stocks)
        {
            var DatesToCheck = CalcDatesToCheck().Select(x => x.ToString("yyyy - MM - ddTHH:mm: ss.ffffffZ", CultureInfo.InvariantCulture));
            var results = new Dictionary<Stock, List<PriceValues>>();
            

            foreach (Stock DanishStock in stocks)
            {
                Console.WriteLine($"Currently fetching for: '{DanishStock.Description}'");
                results.Add(DanishStock, new List<PriceValues>());
                if (!Directory.Exists(GetPath(DanishStock, dataFolder)))
                {
                    foreach (string date in DatesToCheck)
                    {
                        try
                        {
                            results[DanishStock].AddRange(await saxoDataHandler.GetHistoricData(AssetTypes.Stock, DanishStock.Identifier, date));
                        }
                        catch (Exception)
                        {
                            Console.WriteLine("En exception occured, retrying in 90sec.");
                            Thread.Sleep(100000);
                            results[DanishStock].AddRange(await saxoDataHandler.GetHistoricData(AssetTypes.Stock, DanishStock.Identifier, date));
                            continue;
                        }
                    }
                    await CreateFileForDataPoints(DanishStock, results[DanishStock], dataFolder);
                }
            }
        }

        private IEnumerable<(Company, List<PriceValues>)> ExtractDataFromPath(string basePath)
        {
            List<Company> companyList = new List<Company>();
            List<PriceValues>? prices = new List<PriceValues>();
            Company? company = null;

            var folders = Directory.GetDirectories(basePath);

            foreach (var folder in folders)
            {
                var files = Directory.GetFiles(folder, "*.json", SearchOption.AllDirectories);

                using (StreamReader file = File.OpenText(files[1]))
                {
                    company = JsonConvert.DeserializeObject<Company>(file.ReadToEnd());

                    Console.WriteLine(company.Description);
                }

                using (StreamReader file = File.OpenText(files[0]))
                {
                    prices = JsonConvert.DeserializeObject<List<PriceValues>>(file.ReadToEnd());
                    prices.ForEach(x => x.Identifier = company.Identifier);
                }
                yield return (company, prices);
            }
        }


        private async Task CreateFileForDataPoints(Stock stock, List<PriceValues> prices, string dataFolder)
        {
            string path = GetPath(stock, dataFolder);
            string stockText = JsonConvert.SerializeObject(stock);
            string pricesText = JsonConvert.SerializeObject(prices);

            Directory.CreateDirectory(path);

            await File.WriteAllTextAsync(path + "/stock.json", stockText);
            await File.WriteAllTextAsync(path + "/prices.json", pricesText);
        }

        private static string GetPath(Stock stock, string dataFolder)
        {
            return dataFolder + stock.Description;
        }

        private List<DateTime> CalcDatesToCheck(int years = 2, int Horizon = 1, int Count = 1200)
        {
            var To = DateTime.UtcNow;
            var From = To.AddYears(-years);
            double hours = ((Horizon * Count) / 60);
            List<DateTime> result = new List<DateTime>();
            double totalHours = (To - From).TotalHours;
            DateTime current = To;
            for (int i = 0; i < (totalHours / hours); i++)
            {
                current = current.AddHours(-hours);
                result.Add(current);
            }
            return result;
        }
    }
}
