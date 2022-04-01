using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharedObjects
{
    public class Token
    {
        public string value { get; set; }
        public DateTime valid_from { get; set; }
        public DateTime valid_to { get; set; }
    }
}
