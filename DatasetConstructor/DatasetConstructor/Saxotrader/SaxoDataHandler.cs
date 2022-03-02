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
        string token;

        public SaxoDataHandler(string token) 
        {
            this.token = token;
        }
        public async Task<string> GetUserData() 
        {//Checks user data
            var client = new RestClient("https://gateway.saxobank.com/sim/openapi/port/v1/users/me");
            var request = new RestRequest();
            request.AddHeader("Authorization", $"Bearer {token}");
            request.AddHeader("Cookie", "oa-V4_ENT_DMZ_SIM_OA_CORE_8080=DMBEHEAK");
            RestResponse response = await client.ExecuteAsync(request);
            return response.Content;
        }

        public async Task<string> GetCompanyData(string exchange, string keyword, string assetType) 
        {//Returns a single company
            var client = new RestClient($"https://gateway.saxobank.com/sim/openapi/ref/v1/instruments?ExchangeId={exchange}&Keywords={keyword}&AssetTypes={assetType}");
            var request = new RestRequest();
            request.AddHeader("Authorization", $"Bearer {token}");
            request.AddHeader("Cookie", "oa-V4_ENT_DMZ_SIM_OA_CORE_8080=DMBEHEAK");
            RestResponse response = await client.ExecuteAsync(request);
            return response.Content;
        }

        public async Task<StockData> GetCompanyData(string exchange, string assetType) 
        {//Returns a list of companyes that match the exchange and assetType
            var client = new RestClient($"https://gateway.saxobank.com/sim/openapi/ref/v1/instruments?ExchangeId={exchange}&AssetTypes={assetType}");
            var request = new RestRequest();
            request.AddHeader("Authorization", $"Bearer {token}");
            request.AddHeader("Cookie", "oa-V4_ENT_DMZ_SIM_OA_CORE_8080=DMBEHEAK");
            RestResponse response = await client.ExecuteAsync(request);
            return JsonConvert.DeserializeObject<StockData>(response.Content);
        }

        public async Task<DataPoints> GetHistoricData(string assetType,int id ,int Horizon = 1, int count = 1200) 
        {
            var client = new RestClient($"https://gateway.saxobank.com/sim/openapi/chart/v1/charts/?AssetType={assetType}&Count={count}&Horizon={Horizon}&Uic={id}");
            var request = new RestRequest();
            request.AddHeader("Authorization", $"Bearer {token}");
            request.AddHeader("Cookie", "oa-V4_ENT_DMZ_SIM_OA_CORE_8080=DJCEHEAK");
            RestResponse response = await client.ExecuteAsync(request);
            return JsonConvert.DeserializeObject<DataPoints>(response.Content);
        }

    }
}
