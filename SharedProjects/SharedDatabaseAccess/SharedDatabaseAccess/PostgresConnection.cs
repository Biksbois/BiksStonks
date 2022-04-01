using Dapper;
using DatasetConstructor.Saxotrader.Models;
using Npgsql;
using System;
using System.Data;
using System.Linq;

namespace SharedDatabaseAccess
{
    static class PostgresConnection
    {

        public static List<T> GetRowsWithParameters<T>(string query, string connectionString, object param)
        {
            // SELECT * FROM stock WHERE Time < @start
            // connectino.Query<T>(query, new { start })

            using (var connection = new NpgsqlConnection(connectionString))
            {
                connection.Open();
                var value = connection.Query<T>(query, param).ToList();

                return value;
            }
        }

        public static T GetSingleRowWithParameters<T>(string query, string connectionString, object param)
        {
            using (var connection = new NpgsqlConnection(connectionString))
            {
                connection.Open();
                var value = connection.QueryFirstOrDefault<T>(query, param);

                return value;
            }
        }

        public static List<T> GetRows<T>(string query, string connectionString)
        {
            using (var connection = new NpgsqlConnection(connectionString))
            {
                connection.Open();
                var value = connection.Query<T>(query).ToList();

                return value;
            }
        }

        public static void InsertRow(object param, string connectionString, string query)
        {
            using (var connection = new NpgsqlConnection(connectionString))
            {
                connection.Open();
                connection.Query(query, param);
            }
        }


        public static void InsertRows(object param, string connectionString, string query)
        {
            NpgsqlConnection.GlobalTypeMapper.MapComposite<PriceValues>("stock_type");
            NpgsqlConnection.GlobalTypeMapper.MapComposite<Company>("dataset_type");

            using (var connection = new NpgsqlConnection(connectionString))
            {
                connection.Open();
                connection.Query(query, param, commandType:CommandType.StoredProcedure);
            }
        }

        public static void InsertRows<T>(string query, string connectionString, List<T> parameter)
        {
            using (var connection = new NpgsqlConnection(connectionString))
            {
                connection.Open();
                connection.Execute(query, parameter);
            }
        }
    }
}
