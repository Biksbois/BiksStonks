using HtmlAgilityPack;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharedObjects
{
    public static class CompanyAndCategory
    {
        public static Dictionary<string, (string, string)> Values = new Dictionary<string, (string PrimaryCategory, string SecondaryCategory)>()
        {
            { "Blue Vision A/S A", ("finance", "investing") },
            { "Boozt AB", ("detail", "store") },
            { "Brd. Klee B A/S", ("industry", "hardware_store") },
            { "Brdr. A & O Johansen præf. A/S", ("industry", "hardware_store") },
            { "Brdr.Hartmann A/S", ("industry", "factory") },
            { "Brøndbyernes IF Fodbold A/S", ("entertainment", "sport") },
            { "Carlsberg A A/S", ("detail", "brewery") },
            { "Carlsberg B A/S", ("detail", "brewery") },
            { "ChemoMetec A/S", ("medicin", "equipment") },
            { "Chr. Hansen Holding A/S", ("medicin", "biotech") },
            { "Coloplast B A/S", ("medicin", "equipment") },
            { "Columbus A/S", ("it", "consultant") },
            { "Copenhagen Capital A/S", ("finance", "investing") },
            { "D/S Norden", ("industry", "logistic") },
            { "Danske Andelskassers Bank A/S", ("finance", "bank") },
            { "Dantax A/S", ("detail", "tech") },
            { "DFDS A/S", ("industry", "logistic") },
            { "Djurslands Bank A/S", ("finance", "bank") },
            { "EAC Invest A/S", ("finance", "investing") },
            { "Flugger Group A/S", ("industry", "hardware_store") },
            { "GN Store Nord A/S", ("medicin", "equipment") },
            { "Green Hydrogen Systems A/S", ("industry", "energy") },
            { "H+H International A/S", ("industry", "hardware_store") },
            { "H. Lundbeck A/S", ("medicin", "drugs") },
            { "Jyske Bank A/S", ("finance", "bank") },
            { "Matas A/S", ("detail", "store") },
            { "Netcompany Group A/S", ("it", "consultant") },
            { "NKT A/S", ("industry", "factory") },
            { "Nordea Bank Abp", ("finance", "bank") },
            { "A.P. Møller - Mærsk A A/S", ("industry", "logistic") },
            { "A.P. Møller - Mærsk B A/S", ("industry", "logistic") },
            { "Aalborg Boldspilklub A/S", ("entertainment", "sport") },
            { "Agat Ejendomme", ("industry", "construction") },
            { "AGF A/S", ("entertainment", "sport") },
            { "ALK-Abelló B A/S", ("medicin", "drugs") },
            { "Alm. Brand A/S", ("finance", "ensurance") },
            { "Ambu A/S", ("medicin", "equipment") },
            { "Atlantic Petroleum P/F", ("industry", "commodities") },
            { "Bang & Olufsen Holding A/S", ("detail", "tech") },
            { "BankNordik P/F", ("finance", "bank") },
            { "Bavarian Nordic A/S", ("medicin", "drugs") },
            { "BioPorto A/S", ("medicin", "drugs") },
            { "cBrain A/S", ("it", "end-to-end") },
            { "Cemat A/S", ("finance", "real_estate") },
            { "Danske Bank A/S", ("finance", "bank") },
            { "Demant A/S", ("medicin", "equipment") },
            { "DSV A/S", ("industry", "logistic") },
            { "Fast Ejendom Danmark A/S", ("finance", "real_estate") },
            { "FirstFarms A/S", ("industry", "commodities") },
            { "FLSmidth & Co. A/S", ("industry", "factory") },
            { "Novo Nordisk B A/S", ("medicin", "drugs") },
            { "Novozymes A/S", ("medicin", "biotech") },
            { "NTG Nordic Transport Group", ("industry", "logistic") },
            { "Orphazyme A/S", ("medicin", "drugs") },
            { "Pandora A/S", ("detail", "store") },
            { "Rockwool International A A/S", ("industry", "factory") },
            { "Rockwool International B A/S", ("industry", "factory") },
            { "Royal Unibrew A/S", ("detail", "brewery") },
            { "SAS AB", ("industry", "logistic") },
            { "Solar B A/S", ("industry", "factory") },
            { "Spar Nord Bank A/S", ("finance", "bank") },
            { "Sydbank A/S", ("finance", "bank") },
            { "Topdanmark A/S", ("finance", "ensurance") },
            { "TORM Plc A", ("industry", "logistic") },
            { "Tryg A/S", ("finance", "ensurance") },
            { "Vestas Wind Systems A/S", ("industry", "energy") },
            { "Zealand Pharma A/S", ("medicin", "drugs") },
            { "Ørsted A/S", ("industry", "energy") },
            { "Fynske Bank A/S", ("finance", "bank") },
            { "Gabriel Holding A/S", ("industry", "factory") },
            { "Genmab A/S", ("medicin", "drugs") },
            { "German High Street Properties A/S", ("finance", "real_estate") },
            { "Glunz & Jensen Holding A/S", ("industry", "factory") },
            { "GreenMobility A/S", ("industry", "energy") },
            { "Grønlandsbanken A/S", ("finance", "bank") },
            { "Gyldendal B A/S", ("detail", "store") },
            { "Harboes Bryggeri B A/S", ("detail", "brewery") },
            { "HusCompagniet", ("finance", "real_estate") },
            { "Hvidbjerg Bank A/S", ("finance", "bank") },
            { "InterMail A/S", ("it", "end-to-end") },
            { "Jeudan A/S", ("finance", "real_estate") },
            { "Kreditbanken A/S", ("finance", "bank") },
            { "Københavns Lufthavne A/S", ("industry", "logistic") },
            { "Lollands Bank A/S", ("finance", "bank") },
            { "Luxor B A/S", ("finance", "investing") },
            { "Lån og Spar Bank A/S", ("finance", "bank") },
            { "MT Hojgaard Holding A/S", ("industry", "construction") },
            { "Møns Bank A/S", ("finance", "bank") },
            { "Newcap Holding A/S", ("finance", "ensurance") },
            { "Nilfisk Holding A/S", ("detail", "tech") },
            { "NNIT A/S", ("it", "consultant") },
            { "Nordfyns Bank A/S", ("finance", "bank") },
            { "Nordic Shipholding A/S", ("industry", "logistic") },
            { "North Media A/S", ("it", "media_group") },
            { "NTR Holding B A/S", ("industry", "factory") },
            { "Onxeo SA (Copenhagen)", ("medicin", "drugs") },
            { "Park Street A/S", ("finance", "real_estate") },
            { "PARKEN Sport & Entertainment A/S", ("entertainment", "parks") },
            { "Per Aarsleff Holding A/S B", ("industry", "construction") },
            { "Prime Office A/S", ("finance", "real_estate") },
            { "Q-Interline A/S", ("it", "end-to-end") },
            { "Rias B A/S", ("industry", "factory") },
            { "Ringkjøbing Landbobank A/S", ("finance", "bank") },
            { "Roblon B A/S", ("industry", "factory") },
            { "Rovsing A/S", ("it", "end-to-end") },
            { "RTX A/S", ("detail", "tech") },
            { "Sanistål A/S", ("industry", "factory") },
            { "Scandinavian Brake Systems A/S", ("industry", "factory") },
            { "Scandinavian Investment Group A/S", ("finance", "investing") },
            { "Schouw & Co. A/S", ("industry", "factory") },
            { "Silkeborg IF Invest A/S", ("entertainment", "sport") },
            { "SimCorp A/S", ("it", "end-to-end") },
            { "SKAKO A/S", ("industry", "factory") },
            { "Skjern Bank A/S", ("finance", "bank") },
            { "SP Group A/S", ("industry", "factory") },
            { "Sparekassen Sjælland-Fyn A/S", ("finance", "bank") },
            { "Stenocare A/S", ("medicin", "drugs") },
            { "Strategic Investments A/S", ("finance", "investing") },
            { "The Drilling Company of 1972 A/S", ("industry", "commodities") },
            { "Tivoli A/S", ("entertainment", "parks") },
            { "Totalbanken A/S", ("finance", "bank") },
            { "Trifork Holding AG", ("it", "consultant") },
            { "United International Enterprises Ltd", ("finance", "investing") },
            { "Vestjysk Bank A/S", ("finance", "bank") },
            { "Wealth Invest Secure Select Aktier", ("finance", "investing") },
            { "Össur hf.", ("medicin", "equipment" )}
        };

        public static string defaultCategory = "UNKNOWN";

        public static (string primary, string secondary) GetCategoryOrDefault(string companyName)
        {
            if (Values.TryGetValue(companyName, out (string primary, string secondary) categoires))
            {
                return (categoires.primary, categoires.secondary);
            }
            else
            {
                return (defaultCategory, defaultCategory);
            }
        }

        public async static Task<(string,string)> GetCompanyCategory(string tradingSymbol) 
        {
            var url = $"https://www.marketwatch.com/investing/stock/{tradingSymbol.ToLower()}/company-profile?countrycode=dk&mod=mw_quote_tab";
            var httpClient = new HttpClient();
            var html = await httpClient.GetStringAsync(url);
            var htmlDocument = new HtmlDocument();
            htmlDocument.LoadHtml(html);

            var industry = htmlDocument.DocumentNode.SelectSingleNode("/html/body/div[3]/div[7]/div[1]/div[1]/div/ul/li[1]/span");
            var sector = htmlDocument.DocumentNode.SelectSingleNode("/html/body/div[3]/div[7]/div[1]/div[1]/div/ul/li[2]/span");
            return (industry.InnerText,sector.InnerText);
        }
    }
}
