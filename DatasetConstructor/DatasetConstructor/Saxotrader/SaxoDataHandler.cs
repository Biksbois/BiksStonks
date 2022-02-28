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
        {
            var client = new RestClient("https://gateway.saxobank.com/sim/openapi/port/v1/users/me");
            var request = new RestRequest();
            request.AddHeader("Authorization", $"Bearer {token}");
            request.AddHeader("Cookie", "oa-V4_ENT_DMZ_SIM_OA_CORE_8080=DMBEHEAK");
            RestResponse response = await client.ExecuteAsync(request);
            return response.Content;
        }

        public async Task GetCompanyData() 
        {
            var client = new RestClient("https://gateway.saxobank.com/sim/openapi/ref/v1/instruments?ExchangeId=NYSE&Keywords=Coca Cola co&AssetTypes=Stock");
            var request = new RestRequest();
            request.AddHeader("Authorization", $"Bearer {token}");
            request.AddHeader("Cookie", "oa-V4_ENT_DMZ_SIM_OA_CORE_8080=DMBEHEAK");
            RestResponse response = await client.ExecuteAsync(request);
            Console.WriteLine(response.Content);
        }
    }
}
