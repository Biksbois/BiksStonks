﻿using DatasetConstructor.Saxotrader.Models;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Dapper;
using System.Linq;

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
    }
}
