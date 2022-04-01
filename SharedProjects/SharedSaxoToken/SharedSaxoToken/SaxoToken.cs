using SharedDatabaseAccess;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharedSaxoToken
{
    public static class SaxoToken
    {
        public static async Task<string> GetAsync(string username, string password, string edgeLocation, string connectionString, StonksDbConnection stonksdb)
        {
            if (DatabaseToken.TryGet(stonksdb, connectionString, out var token))
            {
                return token;
            }
            else
            {
                token =  await SeleniumDriver.GetToken(username, password, edgeLocation);
                stonksdb.InsertToken(token, connectionString);
                return token;
            }
        }
    }
}
