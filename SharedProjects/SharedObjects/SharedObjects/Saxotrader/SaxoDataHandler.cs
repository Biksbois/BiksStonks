using DatasetConstructor.Saxotrader.Models;
using Newtonsoft.Json;
using RestSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DatasetConstructor.Saxotrader
{
    public class SaxoDataHandler
    {
        public async Task<string> GetUserData(string token) 
        {//Checks user data
            var client = new RestClient("https://gateway.saxobank.com/sim/openapi/port/v1/users/me");
            var request = new RestRequest();
            request.AddHeader("Authorization", $"Bearer {token}");
            request.AddHeader("Cookie", "oa-V4_ENT_DMZ_SIM_OA_CORE_8080=DMBEHEAK");
            RestResponse response = await client.ExecuteAsync(request);
            return response.Content;
        }

        public async Task<string> GetCompanyData(string token, string exchange, string keyword, string assetType) 
        {//Returns a single company
            var client = new RestClient($"https://gateway.saxobank.com/sim/openapi/ref/v1/instruments?ExchangeId={exchange}&Keywords={keyword}&AssetTypes={assetType}");
            var request = new RestRequest();
            request.AddHeader("Authorization", $"Bearer {token}");
            request.AddHeader("Cookie", "oa-V4_ENT_DMZ_SIM_OA_CORE_8080=DMBEHEAK");
            RestResponse response = await client.ExecuteAsync(request);
            return response.Content;
        }

        public async Task<List<Stock>> GetAllCompanyData(string token, string exchange, string assetType)
        {
            var stockData = new StockData();
            var companyInfo = new List<Stock>();
            var isFirst = true;
            var nextUrl = "";

            RestClient client;

            var request = new RestRequest();
            request.AddHeader("Authorization", $"Bearer {token}");
            request.AddHeader("Cookie", "oa-V4_ENT_DMZ_SIM_OA_CORE_8080=DMBEHEAK");

            do
            {
                if (isFirst)
                {
                    client = new RestClient($"https://gateway.saxobank.com/sim/openapi/ref/v1/instruments?ExchangeId={exchange}&AssetTypes={assetType}");
                    isFirst = false;
                }
                else
                {
                    nextUrl = nextUrl.Replace(":443", "");
                    client = new RestClient(nextUrl);
                }

                RestResponse response = await client.ExecuteAsync(request);

                var responseData = JsonConvert.DeserializeObject<StockData>(response.Content);

                nextUrl = responseData.__next;
                companyInfo.AddRange(responseData.Data);

            } while (!String.IsNullOrEmpty(nextUrl));

            return companyInfo;
        }

        public async Task<StockData> GetCompanyData(string token, string exchange, string assetType) 
        {//Returns a list of companyes that match the exchange and assetType
            var client = new RestClient($"https://gateway.saxobank.com/sim/openapi/ref/v1/instruments?ExchangeId={exchange}&AssetTypes={assetType}");
            var request = new RestRequest();
            request.AddHeader("Authorization", $"Bearer {token}");
            request.AddHeader("Cookie", "oa-V4_ENT_DMZ_SIM_OA_CORE_8080=DMBEHEAK");
            RestResponse response = await client.ExecuteAsync(request);
            return JsonConvert.DeserializeObject<StockData>(response.Content);
        }

        public async Task<List<PriceValues>> GetHistoricData(string token, string assetType,int id , string time ,string mode = "from",int Horizon = 1, int count = 1200)
        {//https://gateway.saxobank.com/sim/openapi/chart/v1/charts/?AssetType=Stock&Horizon=1&Mode=from&Time=2022 - 03 - 02T13:03: 48.442486Z&Uic=15611
            var client = new RestClient($"https://gateway.saxobank.com/sim/openapi/chart/v1/charts/?AssetType={assetType}&Count={count}&Horizon={Horizon}&Mode={mode}&Time={time}&Uic={id}");
            var request = new RestRequest();
            request.AddHeader("Authorization", $"Bearer {token}");
            request.AddHeader("Cookie", "oa-V4_ENT_DMZ_SIM_OA_CORE_8080=DJCEHEAK");
            RestResponse response = await client.ExecuteAsync(request);

            if (String.IsNullOrEmpty(response.Content))
            {
                Console.WriteLine($"company '{id}' did not have daat for time {time}");
                return new List<PriceValues>();
            }
            else
            {
                return JsonConvert.DeserializeObject<DataPoints>(response.Content).Data;
            }
        }

    }
}
