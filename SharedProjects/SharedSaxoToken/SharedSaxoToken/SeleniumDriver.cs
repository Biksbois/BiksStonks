using OpenQA.Selenium;
using OpenQA.Selenium.Edge;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharedSaxoToken
{
    public static class SeleniumDriver
    {
        private static string _dropdownButtonXPath = "/html/body/header/div/div[2]/div[3]/div";
        private static string _tokenButtonXPaht = "/html/body/div[3]/div/div[1]/span[7]/a";
        private static string _usernameXPath = "/html/body/div/form/div/div[2]/div[1]/div[2]/input";
        private static string _passwordXPath = "/html/body/div/form/div/div[2]/div[1]/div[3]/input";
        private static string _loginXPath = "/html/body/div/form/div/div[2]/div[1]/div[5]/input";
        private static string _acceptTermsXPath = "/html/body/div[5]/div/main/section/form/section[3]/button";
        private static string _tokenId = "token";

        private static EdgeDriver GetEdgeDriver(string msedgedriverlocation)
        {
            var options = new EdgeOptions();

            options.AddArguments(new List<string>() { "no-sandbox", "headless", "disable-gpu", "window-size=1200,1100" });

            var driver = new EdgeDriver(msedgedriverlocation, options);

            return driver;
        }

        public static async Task<string> GetToken(string username, string password, string msedgedriverLocaton)
        {
            try
            {
                var driver = GetEdgeDriver(msedgedriverLocaton);

                driver.Url = "https://www.developer.saxo/";

                // Open dropdown menu and press "get 24h token
                driver.FindElement(By.XPath(_dropdownButtonXPath)).Click();
                await Task.Delay(1000);

                driver.FindElement(By.XPath(_tokenButtonXPaht)).Click();
                await Task.Delay(1000);


                // Login to saxo with username and email
                var emailBox = driver.FindElement(By.XPath(_usernameXPath));
                var passwordBox = driver.FindElement(By.XPath(_passwordXPath));

                emailBox.SendKeys(username);
                passwordBox.SendKeys(password);

                driver.FindElement(By.XPath(_loginXPath)).Click();

                // Accept terms and conditions
                await Task.Delay(1000);
                driver.FindElement(By.XPath(_acceptTermsXPath)).Click();

                // Copy token
                await Task.Delay(1000);

                var token = driver.FindElement(By.Id(_tokenId)).Text;

                driver.Close();

                // Close browser
                return token;
            }
            catch (Exception e)
            {
                Console.WriteLine($"ERROR WHEN GETTING TOKEN: {e.Message}");
                throw;
            }
        }
    }
}
