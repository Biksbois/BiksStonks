using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.WebSockets;
using System.Text;
using System.Threading.Tasks;
using Serilog;

using ILogger = Serilog.ILogger;

namespace SaxoRealTimeWorker
{
    public class SaxoWebSocket : IDisposable
    {
        private bool _disposed = false;
        private ClientWebSocket _clientWebSocket;
        private readonly ILogger _logger;

        public SaxoWebSocket(ILogger logger)
        {
            _logger = logger;
        }

        public void Dispose()
        {
            _disposed = true;
        }

        internal Task StopWebSocket()
        {
            throw new NotImplementedException();
        }

        internal async Task RunSample(CancellationToken stoppingToken)
        {
            Task taskStartWebSocket = new Task(async () => { await StartWebSocket(); }, stoppingToken);
            taskStartWebSocket.Start();

            Task taskCreateSubscription = new Task(async () => { await CreateSubscription(); }, stoppingToken);
            taskCreateSubscription.Start();

            DateTime tokenDummyExpiryTime = DateTime.Now.AddHours(2);

            Task taskReauthorization = new Task(async () => { await ReauthorizeWhenNeeded(tokenDummyExpiryTime, stoppingToken); }, stoppingToken);
            taskReauthorization.Start();

            Task[] tasks = { taskStartWebSocket, taskCreateSubscription, taskReauthorization };
            try
            {
                Task.WaitAll(tasks, stoppingToken);
            }
            catch (OperationCanceledException)
            {
                return;
            }

            if (!stoppingToken.IsCancellationRequested) 
                _logger.Information("{time} - Listening on web socket. {category}", DateTime.UtcNow, "Listening ");

            //Let's wait until someone stops the sample.
            while (!stoppingToken.IsCancellationRequested)
            {
                await Task.Delay(TimeSpan.FromSeconds(1));
            }
        }

        private Task ReauthorizeWhenNeeded(DateTime tokenDummyExpiryTime, CancellationToken token)
        {
            throw new NotImplementedException();
        }

        private Task CreateSubscription()
        {
            throw new NotImplementedException();
        }

        private Task StartWebSocket()
        {
            throw new NotImplementedException();
        }
    }
}
