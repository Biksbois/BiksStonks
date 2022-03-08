using Serilog;

namespace SaxoRealTimeData
{
    public class SaxoWorker : BackgroundService
    {
        private readonly Serilog.ILogger _logger;

        public SaxoWorker(Serilog.ILogger logger)
        {
            _logger = logger;
        }

        protected override Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    _logger.Warning("horse");
                }
                catch (Exception exception)
                {

                    throw;
                }
                
                Task.Delay(10000, stoppingToken);
            }
         
            return Task.CompletedTask;
        }
    }
}
