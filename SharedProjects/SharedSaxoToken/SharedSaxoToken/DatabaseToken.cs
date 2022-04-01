using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharedDatabaseAccess;

namespace SharedSaxoToken
{
    public static class DatabaseToken
    {
        internal static bool TryGet(StonksDbConnection stonksdb, string connectionSting, out string token)
        {
            var tokenRow = stonksdb.GetToken(DateTime.UtcNow, connectionSting);

            if (tokenRow == null)
            {
                token = null;
                return false;
            }
            else
            {
                token = tokenRow.value;
                return true;
            }
        }
    }
}
