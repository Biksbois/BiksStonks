using DatasetConstructor.Saxotrader.Models;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Dapper;
using System.Linq;
using SharedObjects;

namespace SharedDatabaseAccess
{
    public class StonksDbConnection
    {
        public void InsertStocks(List<PriceValues> stocks, string connectionString)
        {
            var query = "upsert_stock";

            var param = new { source = stocks.ToArray() };

            PostgresConnection.InsertRows(param, connectionString, query);
        }

        public void InsertCompanies(List<Company> companies, string connectionString)
        {
            var query = "upsert_dataset";

            var param = new { source = companies.ToArray() };

            PostgresConnection.InsertRows(param, connectionString, query);
        }

        public Token GetToken(DateTime time, string connectionString)
        {
            var query = "SELECT * FROM token WHERE @time BETWEEN valid_from AND valid_to ORDER BY valid_from LIMIT 1";
            var parameter = new { time };

            var token = PostgresConnection.GetSingleRowWithParameters<Token>(query, connectionString, parameter);

            return token;

        }

        public void InsertToken(string value, string connectionString)
        {
            var query = "INSERT INTO token(value, valid_from, valid_to) VALUES(@value, @valid_from, @valid_to);";
            
            var valid_from = DateTime.UtcNow;
            var valid_to = valid_from.AddHours(20);
            
            var param = new { value, valid_from, valid_to };

            PostgresConnection.InsertRow(param, connectionString, query);
        }

    }
}
