using Serilog;
using ILogger = Serilog.ILogger;

namespace SaxoRealTimeWorker
{
    public class Worker : BackgroundService
    {
        private readonly ILogger _logger;
        private static Action _closeConnectionAction;

        public Worker(ILogger logger)
        {
            _logger = logger;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.Information("{time} - SaxoRealTimeWorker is about to start {category}", DateTime.UtcNow, "start");

            var saxoSocket = new SaxoWebSocket();

            async void CloseConnectionCallback() => await saxoSocket.StopWebSocket();
            _closeConnectionAction = CloseConnectionCallback;

            try
            {
                await saxoSocket.RunSample(stoppingToken);
            }
            catch (Exception e)
            {
                _logger.Error("{time} - exception occured {exception} {category}", DateTime.UtcNow, e.Message, "exception");
                throw;
            }
            finally
            {
                saxoSocket.Dispose();
            }

            _logger.Information("{time} - SaxoRealTimeWorker is about to stop {category}", DateTime.UtcNow, "stop");
        }
    }
}